import { useEffect, useState } from 'react'
import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle() {
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')
  useEffect(() => {
    const root = document.documentElement
    if (dark) {
      root.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [dark])
  return (
    <button onClick={()=>setDark(v=>!v)} className="inline-flex items-center gap-2 border px-2 py-1 rounded text-sm">
      {dark ? <Sun size={16}/> : <Moon size={16}/>}<span>{dark? 'Light':'Dark'}</span>
    </button>
  )
}


