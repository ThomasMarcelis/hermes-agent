import { useCallback, useRef, useState } from 'react'

export function useStash() {
  const stashRef = useRef<string[]>([])
  const [stashCount, setStashCount] = useState(0)

  const pushStash = useCallback((text: string) => {
    if (!text) {
      return false
    }

    stashRef.current.unshift(text)
    setStashCount(stashRef.current.length)

    return true
  }, [])

  /**
   * Cycle the stash queue: take the front item, and if there is text in the
   * composer, push it to the back.  Returns the popped front text or ''.
   */
  const cycleStash = useCallback((currentText: string) => {
    if (stashRef.current.length === 0) {
      return ''
    }

    const text = stashRef.current.shift()!

    if (currentText) {
      stashRef.current.push(currentText)
    }

    setStashCount(stashRef.current.length)

    return text
  }, [])

  const popStashAt = useCallback((index: number) => {
    const arr = stashRef.current

    if (index < 0 || index >= arr.length) {
      return ''
    }

    const [text] = arr.splice(index, 1)
    setStashCount(arr.length)

    return text ?? ''
  }, [])

  const peekStash = useCallback(() => stashRef.current[0] ?? '', [])

  const getStashList = useCallback(() => [...stashRef.current], [])

  const clearStash = useCallback(() => {
    stashRef.current = []
    setStashCount(0)
  }, [])

  return { clearStash, cycleStash, getStashList, peekStash, popStashAt, pushStash, stashCount, stashRef }
}
