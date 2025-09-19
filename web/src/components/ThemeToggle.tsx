import React from 'react'
import { useTheme } from '../contexts/ThemeContext'

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      className="relative p-2 rounded-xl bg-white/80 dark:bg-slate-800/90 backdrop-blur border border-slate-200/50 dark:border-slate-600/50 hover:bg-white dark:hover:bg-slate-700 transition-all duration-300 shadow-sm hover:shadow-md group"
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      <div className="relative w-5 h-5">
        {/* Sun icon */}
        <svg
          className={`absolute inset-0 w-5 h-5 transition-all duration-500 ${
            theme === 'light' 
              ? 'opacity-100 rotate-0 scale-100 text-amber-500' 
              : 'opacity-0 -rotate-90 scale-75 text-amber-300'
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <circle cx="12" cy="12" r="5" />
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
        </svg>
        
        {/* Moon icon */}
        <svg
          className={`absolute inset-0 w-5 h-5 transition-all duration-500 ${
            theme === 'dark' 
              ? 'opacity-100 rotate-0 scale-100 text-blue-300' 
              : 'opacity-0 rotate-90 scale-75 text-blue-500'
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      </div>
      
      {/* Hover effect ring */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-amber-300/20 to-blue-300/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
    </button>
  )
}
