import { useSearchStore } from '../store/searchStore'

export default function ControlsBar() {
  const { lastLimit, useJudge, debug, set } = useSearchStore()
  return (
    <div className="flex flex-wrap gap-4 items-center text-sm">
      <label className="flex items-center gap-2">Limit
        <input type="range" min={6} max={24} value={lastLimit} onChange={(e)=>set({ lastLimit: Number(e.target.value) })} />
        <span>{lastLimit}</span>
      </label>
      <label className="flex items-center gap-2">Judge
        <input type="checkbox" checked={useJudge} onChange={(e)=>set({ useJudge: e.target.checked })} />
      </label>
      <label className="flex items-center gap-2">Debug
        <input type="checkbox" checked={debug} onChange={(e)=>set({ debug: e.target.checked })} />
      </label>
    </div>
  )
}


