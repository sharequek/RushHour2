import React from 'react'
import { ListingCard } from './ListingCard'
import { useInfiniteListings } from '../hooks/useInfiniteListings'

export const ListingGrid: React.FC = () => {
  const { items, hasMore, loading, error, sentinelRef } = useInfiniteListings({ pageSize: 24, sort: 'score', order: 'desc' })

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <header className="py-8 flex items-center justify-between">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-slate-600 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent transition-colors duration-300">
          Top Listings
        </h1>
      </header>

      <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {items.map(item => (
          <ListingCard key={item.id} item={item} />
        ))}
      </div>

      {error && (
        <div className="mt-6 text-center text-red-600 dark:text-red-300 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700/50 rounded-lg px-4 py-3 transition-colors duration-300">{error}</div>
      )}

      <div ref={sentinelRef} className="py-12 flex items-center justify-center">
        {loading && (
          <div className="flex items-center gap-3 text-slate-600 dark:text-slate-200 bg-white/60 dark:bg-slate-800/80 backdrop-blur px-6 py-4 rounded-2xl shadow-sm border border-slate-200/50 dark:border-slate-600/50 transition-colors duration-300">
            <svg className="animate-spin h-5 w-5 text-blue-600 dark:text-blue-300" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
            </svg>
            <span className="font-medium">Loading more…</span>
          </div>
        )}
        {!hasMore && !loading && (
          <div className="text-gray-400 text-sm">You’ve reached the end.</div>
        )}
      </div>
    </div>
  )
}

