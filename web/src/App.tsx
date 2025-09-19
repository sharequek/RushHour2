import React from 'react'
import { ListingGrid } from './components/ListingGrid'
import { ThemeProvider } from './contexts/ThemeContext'
import { ThemeToggle } from './components/ThemeToggle'
import { ScrapingStatus } from './components/ScrapingStatus'

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}

function AppContent() {
  return (
          <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 transition-colors duration-300">
        <nav className="sticky top-0 z-10 backdrop-blur bg-white/80 dark:bg-slate-900/90 border-b border-slate-200/60 dark:border-slate-600/40 shadow-sm transition-colors duration-300">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="font-bold text-2xl bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-300 dark:to-indigo-300 bg-clip-text text-transparent">
              RushHour2
            </div>
            <div className="flex items-center gap-4">
              <ScrapingStatus />
              <ThemeToggle />
            </div>
          </div>
        </nav>
      <main className="py-8">
        <ListingGrid />
      </main>
    </div>
  )
}

