import { useState } from 'react'
import { useSearchStore } from '../store/searchStore'

export default function SearchBar({ onSubmit, label = 'Search' }: { onSubmit: (q: string) => void; label?: string }) {
  const { lastQuery, set } = useSearchStore()
  const [q, setQ] = useState(lastQuery)
  return (
    <form className="flex items-center gap-2" onSubmit={(e)=>{e.preventDefault(); set({ lastQuery: q }); onSubmit(q)}}>
      <input className="border rounded px-3 py-2 w-full" placeholder="Describe what you want..." value={q} onChange={e=>setQ(e.target.value)} />
      <button className="border rounded px-3 py-2" type="submit">{label}</button>
    </form>
  )
}


