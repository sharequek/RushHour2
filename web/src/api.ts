import type { ListingsResponse } from './types'

export interface ScrapingStatus {
  last_scraped: string | null
  total_listings: number
  recent_listings: number
}

const BASE = import.meta.env.VITE_API_URL || '' // use proxy if empty

export async function fetchListings(offset: number, limit: number, sort: string = 'score', order: 'asc' | 'desc' = 'desc'):
  Promise<ListingsResponse> {
  const url = new URL('/api/listings', BASE || window.location.origin)
  url.searchParams.set('offset', String(offset))
  url.searchParams.set('limit', String(limit))
  url.searchParams.set('sort', sort)
  url.searchParams.set('order', order)

  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function fetchScrapingStatus(): Promise<ScrapingStatus> {
  const url = new URL('/api/scraping-status', BASE || window.location.origin)
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

