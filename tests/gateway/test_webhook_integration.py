"""Integration tests for the generic webhook platform adapter.

These tests exercise end-to-end flows through the webhook adapter:
1. GitHub PR webhook → agent MessageEvent created
2. Skills config injects skill content into the prompt
3. Cross-platform delivery routes to a mock Telegram adapter
4. GitHub comment delivery invokes ``gh`` CLI (mocked subprocess)
"""

import asyncio
import base64
import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    """Create a WebhookAdapter with the given routes."""
    extra = {"host": "0.0.0.0", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    """Build the aiohttp Application from the adapter."""
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _github_signature(body: bytes, secret: str) -> str:
    """Compute X-Hub-Signature-256 for *body* using *secret*."""
    return "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


# A realistic GitHub pull_request event payload (trimmed)
GITHUB_PR_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "title": "Add webhook adapter",
        "body": "This PR adds a generic webhook platform adapter.",
        "html_url": "https://github.com/org/repo/pull/42",
        "user": {"login": "contributor"},
        "head": {"ref": "feature/webhooks"},
        "base": {"ref": "main"},
    },
    "repository": {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    },
    "sender": {"login": "contributor"},
}


# ===================================================================
# Test 1: GitHub PR webhook triggers agent
# ===================================================================

class TestGitHubPRWebhook:

    @pytest.mark.asyncio
    async def test_github_pr_webhook_triggers_agent(self):
        """POST with a realistic GitHub PR payload should:
        1. Return 202 Accepted
        2. Call handle_message with a MessageEvent
        3. The event text contains the rendered prompt
        4. The event source has chat_type 'webhook'
        """
        secret = "gh-webhook-test-secret"
        routes = {
            "github-pr": {
                "secret": secret,
                "events": ["pull_request"],
                "prompt": (
                    "Review PR #{number} by {sender.login}: "
                    "{pull_request.title}\n\n{pull_request.body}"
                ),
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        app = _create_app(adapter)
        body = json.dumps(GITHUB_PR_PAYLOAD).encode()
        sig = _github_signature(body, secret)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "gh-delivery-001",
                },
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["status"] == "accepted"
            assert data["route"] == "github-pr"
            assert data["event"] == "pull_request"
            assert data["delivery_id"] == "gh-delivery-001"

        # Let the asyncio.create_task fire
        await asyncio.sleep(0.05)

        assert len(captured_events) == 1
        event = captured_events[0]
        assert "Review PR #42 by contributor" in event.text
        assert "Add webhook adapter" in event.text
        assert event.source.chat_type == "webhook"
        assert event.source.platform == Platform.WEBHOOK
        assert "github-pr" in event.source.chat_id
        assert event.message_id == "gh-delivery-001"


# ===================================================================
# Test 2: Skills injected into prompt
# ===================================================================

class TestSkillsInjection:

    @pytest.mark.asyncio
    async def test_skills_injected_into_prompt(self):
        """When a route has skills: [code-review], the adapter should
        call build_skill_invocation_message() and use its output as the
        prompt instead of the raw template render."""
        routes = {
            "pr-review": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["pull_request"],
                "prompt": "Review this PR: {pull_request.title}",
                "skills": ["code-review"],
            }
        }
        adapter = _make_adapter(routes)

        captured_events: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured_events.append(event)

        adapter.handle_message = _capture

        skill_content = (
            "You are a code reviewer. Review the following:\n"
            "Review this PR: Add webhook adapter"
        )

        # The imports are lazy (inside the handler), so patch the source module
        with patch(
            "agent.skill_commands.build_skill_invocation_message",
            return_value=skill_content,
        ) as mock_build, patch(
            "agent.skill_commands.get_skill_commands",
            return_value={"/code-review": {"name": "code-review"}},
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/webhooks/pr-review",
                    json=GITHUB_PR_PAYLOAD,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-GitHub-Delivery": "skill-test-001",
                    },
                )
                assert resp.status == 202

            await asyncio.sleep(0.05)

            assert len(captured_events) == 1
            event = captured_events[0]
            # The prompt should be the skill content, not the raw template
            assert "You are a code reviewer" in event.text
            mock_build.assert_called_once()


# ===================================================================
# Test 3: Cross-platform delivery (webhook → Telegram)
# ===================================================================

class TestCrossPlatformDelivery:

    @pytest.mark.asyncio
    async def test_cross_platform_delivery(self):
        """When deliver='telegram', the response is routed to the
        Telegram adapter via gateway_runner.adapters."""
        routes = {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "telegram",
                "deliver_extra": {"chat_id": "12345"},
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # Set up a mock gateway runner with a mock Telegram adapter
        mock_tg_adapter = AsyncMock()
        mock_tg_adapter.send = AsyncMock(return_value=SendResult(success=True))

        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.TELEGRAM: mock_tg_adapter}
        mock_runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
        )
        adapter.gateway_runner = mock_runner

        # First, simulate a webhook POST to set up delivery_info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/alerts",
                json={"message": "Server is on fire!"},
                headers={"X-GitHub-Delivery": "alert-001"},
            )
            assert resp.status == 202

        # The adapter should have stored delivery info
        chat_id = "webhook:alerts:alert-001"
        assert chat_id in adapter._delivery_info

        # Now call send() as if the agent has finished
        result = await adapter.send(chat_id, "I've acknowledged the alert.")

        assert result.success is True
        mock_tg_adapter.send.assert_awaited_once_with(
            "12345", "I've acknowledged the alert.", metadata=None
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info


# ===================================================================
# Test 4: GitHub comment delivery via gh CLI
# ===================================================================

class TestGitHubCommentDelivery:

    @pytest.mark.asyncio
    async def test_github_comment_delivery(self):
        """When deliver='github_comment', the adapter invokes
        ``gh pr comment`` via subprocess.run (mocked)."""
        routes = {
            "pr-bot": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Review: {pull_request.title}",
                "deliver": "github_comment",
                "deliver_extra": {
                    "repo": "{repository.full_name}",
                    "pr_number": "{number}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        # POST a webhook to set up delivery info
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/pr-bot",
                json=GITHUB_PR_PAYLOAD,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-GitHub-Delivery": "gh-comment-001",
                },
            )
            assert resp.status == 202

        chat_id = "webhook:pr-bot:gh-comment-001"
        assert chat_id in adapter._delivery_info

        # Verify deliver_extra was rendered with payload data
        delivery = adapter._delivery_info[chat_id]
        assert delivery["deliver_extra"]["repo"] == "org/repo"
        assert delivery["deliver_extra"]["pr_number"] == "42"

        # Mock subprocess.run and call send()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Comment posted"
        mock_result.stderr = ""

        with patch(
            "gateway.platforms.webhook.subprocess.run",
            return_value=mock_result,
        ) as mock_run:
            result = await adapter.send(
                chat_id, "LGTM! The code looks great."
            )

        assert result.success is True
        mock_run.assert_called_once_with(
            [
                "gh", "pr", "comment", "42",
                "--repo", "org/repo",
                "--body", "LGTM! The code looks great.",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Delivery info is retained after send() so interim status messages
        # don't strand the final response (TTL-based cleanup happens on POST).
        assert chat_id in adapter._delivery_info

# ===================================================================
# Test 5: Cloudflare Email Service delivery
# ===================================================================

class TestCloudflareEmailDelivery:

    @pytest.mark.asyncio
    async def test_cloudflare_email_delivery(self, tmp_path):
        """When deliver='cloudflare_email', final agent text is POSTed to
        Cloudflare Email Service with rendered sender/recipient/threading.
        """
        env_file = tmp_path / "cloudflare.env"
        env_file.write_text(
            "CLOUDFLARE_ACCOUNT_ID=acct_123\n"
            "CLOUDFLARE_API_TOKEN=token_456\n",
            encoding="utf-8",
        )
        routes = {
            "agent-email": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{body_text}",
                "deliver": "cloudflare_email",
                "deliver_extra": {
                    "env_file": str(env_file),
                    "from": "{recipient}",
                    "to": "{from.address}",
                    "subject": "{subject}",
                    "in_reply_to": "{message_id}",
                    "references": "{message_id}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()
        payload = {
            "body_text": "Please update the site",
            "recipient": "agent@example.com",
            "from": {"address": "thomas@example.com"},
            "subject": "Website feedback",
            "message_id": "<msg-1@example.com>",
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/agent-email",
                json=payload,
                headers={"X-Request-ID": "email-001"},
            )
            assert resp.status == 202

        calls = []

        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return b'{"success": true, "errors": []}'

        def fake_urlopen(req, timeout=None):
            calls.append((req, timeout))
            return FakeResponse()

        markdown_reply = (
            "# Done\n\n"
            "- **Updated** the site\n"
            "- See [the page](https://example.com/?a=1&b=2)\n\n"
            "<script>alert('x')</script>\n\n"
            "[unsafe](javascript:alert(1))"
        )

        with patch("gateway.platforms.webhook_email.urllib.request.urlopen", fake_urlopen):
            result = await adapter.send(
                "webhook:agent-email:email-001",
                markdown_reply,
            )

        assert result.success is True
        assert len(calls) == 1
        req, timeout = calls[0]
        assert timeout == 45
        assert req.full_url == (
            "https://api.cloudflare.com/client/v4/accounts/acct_123/email/sending/send"
        )
        assert req.headers["Authorization"] == "Bearer token_456"
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["to"] == "thomas@example.com"
        assert sent["from"] == "agent@example.com"
        assert sent["subject"] == "Re: Website feedback"
        assert sent["text"] == markdown_reply
        assert sent["headers"] == {
            "In-Reply-To": "<msg-1@example.com>",
            "References": "<msg-1@example.com>",
        }
        assert "html" in sent
        assert "<h1>Done</h1>" in sent["html"]
        assert "<strong>Updated</strong>" in sent["html"]
        assert 'href="https://example.com/?a=1&amp;b=2"' in sent["html"]
        assert "<script" not in sent["html"].lower()
        assert "javascript:" not in sent["html"].lower()

    @pytest.mark.asyncio
    async def test_agent_email_attachment_bytes_are_saved_and_prompt_augmented(self, tmp_path, monkeypatch):
        """agent_email_received payload attachments are decoded to local files
        and local paths are appended to the agent prompt before dispatch.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        routes = {
            "agent-email": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["agent_email_received"],
                "prompt": "{agent_prompt}",
                "deliver": "log",
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        image_bytes = b"\x89PNG\r\n\x1a\nsmall-test-image"
        payload = {
            "event_type": "agent_email_received",
            "agent_prompt": "Please inspect the attachment.",
            "body_text": "Please inspect the attachment.",
            "recipient": "agent@example.com",
            "from": {"address": "thomas@example.com"},
            "subject": "Attachment test",
            "message_id": "<att-1@example.com>",
            "attachments": {
                "supported": True,
                "count": 1,
                "items": [
                    {
                        "filename": "screenshot.png",
                        "content_type": "image/png",
                        "approx_bytes": len(image_bytes),
                        "content_included": True,
                        "content_encoding": "base64",
                        "content_base64": base64.b64encode(image_bytes).decode("ascii"),
                    }
                ],
            },
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/agent-email",
                json=payload,
                headers={"X-Request-ID": "email-att-001"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        adapter.handle_message.assert_awaited_once()
        await_args = adapter.handle_message.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert isinstance(event, MessageEvent)
        assert "Hermes saved email attachments" in event.text
        assert "screenshot.png" in event.text
        assert "vision_analyze" in event.text

        item = event.raw_message["attachments"]["items"][0]
        assert "content_base64" not in item
        assert item["content_saved"] is True
        saved_path = Path(item["local_path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == image_bytes

    @pytest.mark.asyncio
    async def test_cloudflare_email_intro_reply_uses_payload_reply_plan(self, tmp_path):
        """Introduction handoffs can override the default reply-to-sender rule.

        The authenticated Worker decides the validated recipient set from the
        message To/Cc headers. Hermes must honor that payload-level plan so a
        one-off introduction can continue directly with the introduced address
        while removing the introducer from recipients.
        """
        env_file = tmp_path / "cloudflare.env"
        env_file.write_text(
            "CLOUDFLARE_ACCOUNT_ID=acct_123\n"
            "CLOUDFLARE_API_TOKEN=token_456\n",
            encoding="utf-8",
        )
        routes = {
            "agent-email": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "{agent_prompt}",
                "deliver": "cloudflare_email",
                "deliver_extra": {
                    "env_file": str(env_file),
                    "from": "{recipient}",
                    "to": "{from.address}",
                    "subject": "{subject}",
                    "in_reply_to": "{message_id}",
                    "references": "{message_id}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()
        payload = {
            "event_type": "agent_email_received",
            "agent_prompt": "Private intro prompt\n\nPlease reply to the copied recipient.",
            "body_text": "Please reply directly to the copied recipient and remove me.",
            "recipient": "agent-intro@example.com",
            "from": {"address": "thomas@example.com"},
            "subject": "Intro to project contact",
            "message_id": "<intro-1@example.com>",
            "reply_to": {
                "mode": "introduction",
                "to": ["anton@example.com"],
                "cc": [],
                "excluded": ["agent-intro@example.com", "sender@example.com"],
            },
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/agent-email",
                json=payload,
                headers={"X-Request-ID": "email-intro-001"},
            )
            assert resp.status == 202

        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return b'{"success": true, "errors": []}'

        calls = []
        def fake_urlopen(req, timeout=None):
            calls.append((req, timeout))
            return FakeResponse()

        with patch("gateway.platforms.webhook_email.urllib.request.urlopen", fake_urlopen):
            result = await adapter.send(
                "webhook:agent-email:email-intro-001",
                "Thanks - continuing here directly.",
            )

        assert result.success is True
        assert len(calls) == 1
        sent = json.loads(calls[0][0].data.decode("utf-8"))
        assert sent["to"] == "anton@example.com"
        assert "cc" not in sent
        assert sent["from"] == "agent-intro@example.com"
        assert "thomas@example.com" not in json.dumps(sent)
        assert sent["headers"] == {
            "In-Reply-To": "<intro-1@example.com>",
            "References": "<intro-1@example.com>",
        }


# ===================================================================
# Test 6: Operator mirror for supervised agent-email channels
# ===================================================================

class TestOperatorMirror:

    @pytest.mark.asyncio
    async def test_incoming_and_final_outgoing_are_mirrored_to_discord_thread(self, tmp_path):
        """Payload-provided operator_mirror copies inbound/outbound email
        content to Discord while the canonical reply still goes via email.
        """
        env_file = tmp_path / "cloudflare.env"
        env_file.write_text(
            "CLOUDFLARE_ACCOUNT_ID=acct_123\n"
            "CLOUDFLARE_API_TOKEN=token_456\n",
            encoding="utf-8",
        )
        routes = {
            "agent-email": {
                "secret": _INSECURE_NO_AUTH,
                "events": ["agent_email_received"],
                "prompt": "{agent_prompt}",
                "deliver": "cloudflare_email",
                "deliver_extra": {
                    "env_file": str(env_file),
                    "from": "{recipient}",
                    "to": "{from.address}",
                    "subject": "{subject}",
                    "in_reply_to": "{message_id}",
                    "references": "{message_id}",
                },
            }
        }
        adapter = _make_adapter(routes)
        adapter.handle_message = AsyncMock()

        mock_discord_adapter = AsyncMock()
        mock_discord_adapter.send = AsyncMock(return_value=SendResult(success=True))
        mock_runner = MagicMock()
        mock_runner.adapters = {Platform.DISCORD: mock_discord_adapter}
        mock_runner.config = GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")}
        )
        adapter.gateway_runner = mock_runner

        payload = {
            "event_type": "agent_email_received",
            "agent_prompt": "Private project prompt\n\nPlease review this page.",
            "body_text": "Please review this page.",
            "recipient": "agent-project@example.com",
            "channel": {"agent_key": "jd-parigo-anton"},
            "from": {"raw": "Client <client@example.com>", "address": "client@example.com"},
            "subject": "Project website",
            "message_id": "<anton-1@example.com>",
            "trust": {"spf": "pass", "dkim": "pass", "dmarc": "pass"},
            "operator_mirror": {
                "enabled": True,
                "platform": "discord",
                "chat_id": "parent-channel",
                "thread_id": "thread-123",
                "label": "Client / Project",
                "include_incoming": True,
                "include_outgoing": True,
            },
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/agent-email",
                json=payload,
                headers={"X-Request-ID": "email-anton-001"},
            )
            assert resp.status == 202

        await asyncio.sleep(0.05)
        assert mock_discord_adapter.send.await_count == 1
        incoming_call = mock_discord_adapter.send.await_args_list[0]
        assert incoming_call.args[0] == "parent-channel"
        assert "Client / Project - incoming email" in incoming_call.args[1]
        assert "Please review this page." in incoming_call.args[1]
        assert incoming_call.kwargs == {"metadata": {"thread_id": "thread-123"}}

        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return b'{"success": true, "errors": []}'

        with patch("gateway.platforms.webhook_email.urllib.request.urlopen", return_value=FakeResponse()):
            result = await adapter.send(
                "webhook:agent-email:email-anton-001",
                "I reviewed it; here are the project-specific notes.",
                metadata={"notify": True},
            )

        assert result.success is True
        assert mock_discord_adapter.send.await_count == 2
        outgoing_call = mock_discord_adapter.send.await_args_list[1]
        assert outgoing_call.args[0] == "parent-channel"
        assert "Client / Project - outgoing email reply" in outgoing_call.args[1]
        assert "Email delivery:** sent" in outgoing_call.args[1]
        assert "project-specific notes" in outgoing_call.args[1]
        assert outgoing_call.kwargs == {"metadata": {"thread_id": "thread-123"}}
