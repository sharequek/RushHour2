import React from 'react'
import type { Listing } from '../types'
import { PhotoCarousel } from './PhotoCarousel'

function fmtPrice(n: number | null) {
  if (n == null) return '—'
  return `$${n.toLocaleString()}`
}

function fmtBedsBaths(b: number | null, ba: number | null) {
  const parts: string[] = []
  if (b != null) {
    if (b === 0) {
      parts.push('Studio')
    } else {
      parts.push(`${b} bd`)
    }
  }
  if (ba != null) parts.push(`${ba} ba`)
  return parts.join(' · ') || '—'
}

function fmtSqft(s: number | null, conf: number | null) {
  if (!s) return '—'
  const c = conf != null ? ` (${Math.round(conf * 100)}%)` : ''
  return `${s}${c}`
}

function fmtCommute(m: number | null) {
  if (!m) return '—'
  return `${m} min`
}

function fmtScore(s: number | null) {
  if (s == null) return '—'
  return `${s}`
}

function fmtBuildingAddress(listingName: string, buildingAddress: string) {
  if (buildingAddress === '—' || buildingAddress === listingName) return '—'
  
  // Try to extract just the city, state, zip portion
  // Look for common patterns like "Street Name, City, State Zip"
  const parts = buildingAddress.split(', ')
  if (parts.length >= 2) {
    // Remove the first part (street address) and join the rest
    return parts.slice(1).join(', ')
  }
  
  // Fallback: if no comma pattern, try to find city/state patterns
  // Look for patterns like "City, State Zip" or "City State Zip"
  const cityStateMatch = buildingAddress.match(/([A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?|[A-Za-z\s]+\s+[A-Z]{2}\s*\d{5}(?:-\d{4})?)$/)
  if (cityStateMatch) {
    return cityStateMatch[1].trim()
  }
  
  // If we can't parse it, return the original
  return buildingAddress
}

export const ListingCard: React.FC<{ item: Listing }> = ({ item }) => {
  const price = item.effective_price ?? item.price
  const listingName = item.listing_name || '—'
  const buildingAddress = item.building_address || '—'
  const formattedBuildingAddress = fmtBuildingAddress(listingName, buildingAddress)
  const photos = (item.photo_urls && item.photo_urls.length > 0)
    ? item.photo_urls
    : (item.photo_url ? [item.photo_url] : [])

  return (
    <div className="group block rounded-2xl bg-white/80 dark:bg-slate-800/90 backdrop-blur shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden border border-slate-200/50 dark:border-slate-600/50 hover:border-slate-300/70 dark:hover:border-slate-500/70 hover:-translate-y-1">
      <PhotoCarousel photos={photos} alt={listingName} className="group" />
      <div className="p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <a 
              href={item.url} 
              target="_blank" 
              rel="noreferrer"
              className="block text-lg font-semibold text-slate-800 dark:text-slate-100 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-200"
              title={listingName}
            >
              {listingName}
            </a>
            {formattedBuildingAddress !== '—' && (
              <a 
                href={item.url} 
                target="_blank" 
                rel="noreferrer"
                className="block text-sm text-slate-500 dark:text-slate-400 hover:text-blue-500 dark:hover:text-blue-300 transition-colors duration-200 mt-1"
                title={buildingAddress}
              >
                {formattedBuildingAddress}
              </a>
            )}
          </div>
          <div className="shrink-0 inline-flex items-center text-sm px-3 py-1 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 dark:bg-blue-500 text-white font-bold shadow-sm">{fmtScore(item.score)}</div>
        </div>
        <div className="mt-2 text-2xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-300 to-teal-300 bg-clip-text text-transparent">{fmtPrice(price)}</div>
        <div className="mt-2 text-sm text-slate-600 dark:text-slate-200 font-medium">{fmtBedsBaths(item.beds, item.baths)}</div>
        <div className="mt-3 text-sm text-slate-500 dark:text-slate-300 flex gap-4">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-slate-300 dark:bg-slate-400 rounded-full"></span>
            Sqft: {fmtSqft(item.sqft, item.ocr_sqft_confidence)}
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-blue-300 dark:bg-blue-400 rounded-full"></span>
            Commute: {fmtCommute(item.commute_min)}
          </span>
        </div>
      </div>
    </div>
  )
}

