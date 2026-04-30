import { Text } from '@hermes/ink'

import { isMac } from '../lib/platform.js'
import type { Theme } from '../theme.js'

export function StashIndicator({ count, t, textInPrompt }: { count: number; t: Theme, textInPrompt: boolean }) {
  if (!count) {
    return null
  }

  const mod = isMac ? 'Cmd' : 'Ctrl'

  return (
    <Text color={t.color.accent} dimColor>
      {`${count} stashed message${count === 1 ? '' : 's'} ${textInPrompt? `\u00b7 ${mod}+S to stash ` : ''}\u00b7 ${mod}+P to ${textInPrompt? 'pop': 'cycle'}`}
    </Text>
  )
}
