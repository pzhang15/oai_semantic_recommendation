import { useMutation } from '@tanstack/react-query'
import { postOutfit } from '../lib/api'
import SearchBar from '../components/SearchBar'
import { useSearchStore } from '../store/searchStore'
import OutfitSlots from '../components/OutfitSlots'
import DebugTrace from '../components/DebugTrace'

export default function OutfitPage() {
  const { useJudge, debug } = useSearchStore()
  const m = useMutation({ mutationFn: (q: string)=>postOutfit({ query: q, use_judge: useJudge, debug }) })
  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-4">
      <SearchBar onSubmit={(q)=>m.mutate(q)} label="Compose Outfit" />
      {m.isPending && <div>Composing...</div>}
      {m.error && <div className="text-red-600 text-sm">Error: {(m.error as any).message}</div>}
      {m.data && (
        <>
          <div className="flex items-center justify-between">
            <div className="text-sm">{m.data.summary}</div>
            <div className="text-sm">Total: ${m.data.total_price.toFixed(2)} {m.data.under_budget ? <span className="text-green-600">Under budget</span> : <span className="text-red-600">Over budget</span>}</div>
          </div>
          <OutfitSlots data={m.data} />
          <div className="text-sm text-neutral-600 dark:text-neutral-300">{m.data.tips}</div>
          <DebugTrace trace={m.data.trace} />
        </>
      )}
    </div>
  )
}


