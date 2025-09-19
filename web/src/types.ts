export type Listing = {
  id: number
  url: string
  combined_address: string | null
  listing_name: string | null
  building_address: string | null
  price: number | null
  effective_price: number | null
  beds: number | null
  baths: number | null
  sqft: number | null
  ocr_sqft_confidence: number | null
  commute_min: number | null
  score: number | null
  photo_url: string | null
  photo_urls?: string[]
}

export type ListingsResponse = {
  offset: number
  limit: number
  total: number
  items: Listing[]
}
