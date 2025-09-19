import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchListings } from '../api'
import type { Listing } from '../types'

type Options = {
  pageSize?: number
  sort?: string
  order?: 'asc' | 'desc'
}

export function useInfiniteListings(opts: Options = {}) {
  const pageSize = opts.pageSize ?? 24
  const sort = opts.sort ?? 'score'
  const order = opts.order ?? 'desc'

  const [items, setItems] = useState<Listing[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadingRef = useRef(false)

  const loadMore = useCallback(async () => {
    if (loadingRef.current || !hasMore) return
    loadingRef.current = true
    setLoading(true)
    setError(null)
    try {
      const res = await fetchListings(offset, pageSize, sort, order)
      setItems(prev => {
        // Prevent duplicate items by checking if we already have these items
        const newItems = res.items.filter(newItem => 
          !prev.some(existingItem => existingItem.id === newItem.id)
        )
        return [...prev, ...newItems]
      })
      setOffset(prev => prev + res.items.length)
      setHasMore(offset + res.items.length < res.total && res.items.length > 0)
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load')
    } finally {
      setLoading(false)
      loadingRef.current = false
    }
  }, [offset, pageSize, sort, order, hasMore])

  // Initial load
  useEffect(() => {
    // reset when options change
    setItems([])
    setOffset(0)
    setHasMore(true)
    loadingRef.current = false
  }, [pageSize, sort, order])

  // Load first page on mount or on reset
  useEffect(() => {
    if (items.length === 0 && hasMore && !loadingRef.current) {
      loadMore()
    }
  }, [items.length, hasMore, loadMore])

  const sentinelRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const node = sentinelRef.current
    if (!node) return
    const observer = new IntersectionObserver(entries => {
      const [entry] = entries
      if (entry.isIntersecting && hasMore && !loading) {
        loadMore()
      }
    }, { rootMargin: '800px 0px' })
    observer.observe(node)
    return () => observer.disconnect()
  }, [loadMore, hasMore, loading])

  return { items, hasMore, loading, error, loadMore, sentinelRef }
}

