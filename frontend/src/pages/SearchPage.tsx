import { useMutation } from '@tanstack/react-query'
import { useRef, useState } from 'react'
import { postSearch } from '../lib/api'
import { useSearchStore } from '../store/searchStore'
import SearchBar from '../components/SearchBar'
import ProductGrid from '../components/ProductGrid'
import DebugTrace from '../components/DebugTrace'
import ProductModal from '../components/ProductModal'

export default function SearchPage() {
  const { lastQuery } = useSearchStore()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [page, setPage] = useState<number>(1)
  const [useJudge, setUseJudge] = useState<boolean>(true)
  const [resultId, setResultId] = useState<string | undefined>(undefined)
  const t0Ref = useRef<number | null>(null)
  const [e2eMs, setE2eMs] = useState<number | null>(null)
  const m = useMutation({
    mutationFn: ({ q, p, rid }: { q: string; p: number; rid?: string })=>postSearch({ query: q, page: p, limit: 12, use_judge: useJudge, debug: true, result_id: rid }),
    onMutate: ()=>{ t0Ref.current = performance.now(); setE2eMs(null) },
    onSettled: (data)=>{ if (t0Ref.current != null) { setE2eMs(performance.now() - t0Ref.current) } t0Ref.current = null; if (data?.result_id) setResultId(data.result_id) },
  })
  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-4">
      <SearchBar onSubmit={(q)=>{ setPage(1); setResultId(undefined); m.mutate({ q, p: 1 }) }} />
      {/* Header: summary and judge toggle */}
      {m.error && <div className="text-red-600 text-sm">Error: {(m.error as any).message}</div>}
      {/* Header: only when data */}
      {m.data && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-neutral-700 dark:text-neutral-300">{m.data.summary}</div>
          <div className="flex items-center gap-2 text-sm">
            <label className="flex items-center gap-1 mr-2">
              <input type="checkbox" checked={useJudge} disabled={m.isPending} onChange={(e)=>{ setUseJudge(e.target.checked); setPage(1); setResultId(undefined); m.mutate({ q: lastQuery, p: 1 }) }} />
              <span>Use judge</span>
            </label>
          </div>
        </div>
      )}

      {/* Results container with overlay spinner (always mounted) */}
      <div className="relative min-h-[200px]">
        <ProductGrid items={m.data?.items ?? []} onSelect={(it)=>setSelectedId(it.id)} />
        {m.isPending && (
          <div className="absolute inset-0 z-10 bg-white/70 dark:bg-neutral-900/60 flex items-center justify-center">
            <div className="h-10 w-10 border-4 border-neutral-300 dark:border-neutral-700 border-t-blue-500 rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* Pagination bar (only when data) */}
      {m.data && (
        <div className="flex items-center justify-between py-2">
          <div className="text-sm">Page {m.data.page ?? page}</div>
          <div className="flex items-center gap-2">
            <button className="border rounded px-2 py-1 disabled:opacity-50" disabled={page<=1 || m.isPending} onClick={()=>{ const np = Math.max(1, page-1); setPage(np); m.mutate({ q: lastQuery, p: np, rid: resultId }) }}>Prev</button>
            <button className="border rounded px-2 py-1 disabled:opacity-50" disabled={!m.data.has_more || m.isPending} onClick={()=>{ const np = page+1; setPage(np); m.mutate({ q: lastQuery, p: np, rid: resultId }) }}>Next</button>
          </div>
        </div>
      )}

      {m.data && (
        <DebugTrace trace={m.data.trace} e2eMs={e2eMs ?? undefined} />
      )}
      {selectedId && m.data && (
        <ProductModal
          item={m.data.items.find(x=>x.id===selectedId)!}
          onClose={()=>setSelectedId(null)}
        />
      )}
    </div>
  )
}


