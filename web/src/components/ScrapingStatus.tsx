import React, { useEffect, useState } from 'react'
import { fetchScrapingStatus, type ScrapingStatus } from '../api'

export function ScrapingStatus() {
  const [status, setStatus] = useState<ScrapingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadStatus() {
      try {
        setLoading(true)
        const data = await fetchScrapingStatus()
        setStatus(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load status')
      } finally {
        setLoading(false)
      }
    }

    loadStatus()
    // Refresh every 5 minutes
    const interval = setInterval(loadStatus, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="text-sm text-slate-600 dark:text-slate-200 bg-slate-100 dark:bg-slate-800/60 px-3 py-1 rounded-full transition-colors duration-300">
        Loading...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-sm text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/20 px-3 py-1 rounded-full transition-colors duration-300">
        Error loading status
      </div>
    )
  }

  if (!status) {
    return (
      <div className="text-sm text-slate-600 dark:text-slate-200 bg-slate-100 dark:bg-slate-800/60 px-3 py-1 rounded-full transition-colors duration-300">
        No data available
      </div>
    )
  }

  const formatLastScraped = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffHours / 24)
    
    if (diffDays > 0) {
      return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`
    } else {
      const diffMinutes = Math.floor(diffMs / (1000 * 60))
      if (diffMinutes > 0) {
        return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`
      } else {
        return 'Just now'
      }
    }
  }

  return (
    <div className="text-sm text-slate-600 dark:text-slate-200 bg-slate-100 dark:bg-slate-800/60 px-3 py-1 rounded-full transition-colors duration-300">
      Last updated: {formatLastScraped(status.last_scraped)}
    </div>
  )
}
