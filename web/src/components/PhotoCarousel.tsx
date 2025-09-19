import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react'

type Props = {
  photos: string[]
  alt: string
  className?: string
}

export const PhotoCarousel: React.FC<Props> = ({ photos, alt, className = '' }) => {
  const items = useMemo(() => photos.filter(Boolean), [photos])
  const containerRef = useRef<HTMLDivElement>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [touchStart, setTouchStart] = useState<number | null>(null)
  const [touchEnd, setTouchEnd] = useState<number | null>(null)

  // Reset index when photos change
  useEffect(() => {
    setCurrentIndex(0)
  }, [items.length])

  // Update current index based on scroll position
  const updateIndex = useCallback(() => {
    if (!containerRef.current) return
    const container = containerRef.current
    const scrollLeft = container.scrollLeft
    const containerWidth = container.clientWidth
    const newIndex = Math.round(scrollLeft / Math.max(1, containerWidth))
    setCurrentIndex(Math.max(0, Math.min(newIndex, items.length - 1)))
  }, [items.length])

  // Programmatic navigation
  const scrollToIndex = useCallback((index: number) => {
    if (!containerRef.current) return
    const clamped = Math.max(0, Math.min(index, items.length - 1))
    const container = containerRef.current
    const containerWidth = container.clientWidth
    container.scrollTo({ left: clamped * containerWidth, behavior: 'smooth' })
  }, [items.length])

  const goToPrevious = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (currentIndex > 0) scrollToIndex(currentIndex - 1)
  }, [currentIndex, scrollToIndex])

  const goToNext = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (currentIndex < items.length - 1) scrollToIndex(currentIndex + 1)
  }, [currentIndex, items.length, scrollToIndex])

  // Keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      goToPrevious(e as any)
    } else if (e.key === 'ArrowRight') {
      goToNext(e as any)
    }
  }, [goToPrevious, goToNext])

  // Touch event handlers for mobile swiping
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    setTouchEnd(null)
    setTouchStart(e.targetTouches[0].clientX)
  }, [])

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX)
  }, [])

  const handleTouchEnd = useCallback(() => {
    if (!touchStart || !touchEnd) return
    
    const distance = touchStart - touchEnd
    const isLeftSwipe = distance > 50
    const isRightSwipe = distance < -50

    if (isLeftSwipe && currentIndex < items.length - 1) {
      scrollToIndex(currentIndex + 1)
    }
    if (isRightSwipe && currentIndex > 0) {
      scrollToIndex(currentIndex - 1)
    }
  }, [touchStart, touchEnd, currentIndex, items.length, scrollToIndex])

  if (!items.length) {
    return (
      <div className={`aspect-[4/3] bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-600 dark:to-slate-700 grid place-items-center ${className}`}>
        <div className="text-center text-slate-400 dark:text-slate-300">
          <div className="text-2xl mb-1">dY"�</div>
          <div className="text-sm">No photos</div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`relative group ${className}`}
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      {/* Main carousel container (native scroll + snap) */}
      <div
        ref={containerRef}
        className={`
          aspect-[4/3] select-none carousel-container
          overflow-x-auto overflow-y-hidden
        `}
        onScroll={updateIndex}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        style={{
          touchAction: 'pan-x',
          scrollSnapType: 'x mandatory',
          WebkitOverflowScrolling: 'touch',
          overscrollBehaviorX: 'contain'
        }}
      >
        <div className="flex h-full">
          {items.map((src, index) => (
            <div
              key={index}
              className="flex-shrink-0 w-full h-full carousel-item"
              style={{ scrollSnapAlign: 'start' }}
            >
              <img
                src={src}
                alt={`${alt} - Photo ${index + 1}`}
                className="w-full h-full object-cover"
                loading="lazy"
                draggable={false}
                onError={(e) => {
                  // Fallback for broken images
                  const target = e.target as HTMLImageElement
                  target.style.display = 'none'
                }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Navigation arrows - only show if multiple photos */}
      {items.length > 1 && (
        <>
          {/* Previous arrow */}
          <div
            onClick={goToPrevious}
            className={`
              absolute top-1/2 left-4 -translate-y-1/2
              cursor-pointer z-10
              transition-all duration-300 ease-out
              hover:opacity-80 active:opacity-60
              ${currentIndex === 0 ? 'opacity-30 cursor-not-allowed' : ''}
            `}
            style={{ pointerEvents: currentIndex === 0 ? 'none' : 'auto' }}
          >
            <svg
              className="w-6 h-6 text-white drop-shadow-lg"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </div>

          {/* Next arrow */}
          <div
            onClick={goToNext}
            className={`
              absolute top-1/2 right-4 -translate-y-1/2
              cursor-pointer z-10
              transition-all duration-300 ease-out
              hover:opacity-80 active:opacity-60
              ${currentIndex === items.length - 1 ? 'opacity-30 cursor-not-allowed' : ''}
            `}
            style={{ pointerEvents: currentIndex === items.length - 1 ? 'none' : 'auto' }}
          >
            <svg
              className="w-6 h-6 text-white drop-shadow-lg"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                d="M9 5l7 7-7 7"
              />
            </svg>
          </div>

          {/* Photo counter */}
          <div className="absolute top-4 right-4 bg-black/60 text-white text-sm px-3 py-1 rounded-full backdrop-blur-sm z-10 select-none pointer-events-none">
            {currentIndex + 1} / {items.length}
          </div>
        </>
      )}
    </div>
  )
}

