import type { SearchItem } from '../lib/types'

type Props = {
  item: SearchItem
  onClose: () => void
}

export default function ProductModal({ item, onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-white dark:bg-neutral-900 rounded-lg shadow-xl max-w-5xl w-[95vw] max-h-[90vh] overflow-auto">
        <div className="flex flex-col md:flex-row">
          <div className="md:w-1/2 bg-neutral-100 dark:bg-neutral-800">
            {item.image_url ? (
              <img src={item.image_url} alt={item.title} className="w-full h-[60vh] object-cover" />
            ) : (
              <div className="w-full h-[60vh]" />
            )}
          </div>
          <div className="md:w-1/2 p-5 space-y-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold leading-snug">{item.title}</div>
                {item.brand && <div className="text-sm text-neutral-500">{item.brand}</div>}
              </div>
              <div className="flex items-center gap-3">
                {item.price != null && (
                  <div className="px-3 py-1 rounded bg-neutral-100 dark:bg-neutral-800 text-base font-semibold">
                    ${item.price.toFixed(2)}
                  </div>
                )}
                <button onClick={onClose} className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300">✕</button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-neutral-500">Price</div>
                <div>{item.price != null ? `$${item.price.toFixed(2)}` : '—'}</div>
              </div>
              <div>
                <div className="text-neutral-500">Score</div>
                <div>{item.score != null ? item.score.toFixed(3) : '—'}</div>
              </div>
              <div>
                <div className="text-neutral-500">ID</div>
                <div className="break-all">{item.id}</div>
              </div>
              {item.product_url && (
                <div>
                  <div className="text-neutral-500">Link</div>
                  <a className="text-blue-600 hover:underline" href={item.product_url} target="_blank" rel="noreferrer">View product</a>
                </div>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {item.average_rating != null && (
                <div>
                  <div className="text-neutral-500">Rating</div>
                  <div>★ {item.average_rating?.toFixed(1)}{typeof item.rating_number === 'number' && isFinite(item.rating_number) ? ` • ${item.rating_number}` : ''}</div>
                </div>
              )}
              {item.main_category && (
                <div>
                  <div className="text-neutral-500">Main category</div>
                  <div>{item.main_category}</div>
                </div>
              )}
              {item.store && (
                <div>
                  <div className="text-neutral-500">Store</div>
                  <div>{item.store}</div>
                </div>
              )}
              {typeof item.vector_index === 'number' && (
                <div>
                  <div className="text-neutral-500">Vector index</div>
                  <div>{item.vector_index}</div>
                </div>
              )}
            </div>
            {Array.isArray(item.images) && item.images.length > 0 && (
              <div className="space-y-1">
                <div className="text-sm text-neutral-500">Images</div>
                <div className="flex gap-2 overflow-auto">
                  {item.images.slice(0,8).map((url: any, idx: number) => (
                    <img key={idx} src={typeof url === 'string' ? url : (url?.hi_res || url?.large || url?.thumb)} alt="" className="h-16 w-16 object-cover rounded" />
                  ))}
                </div>
              </div>
            )}
            {Array.isArray(item.features) && item.features.length > 0 && (
              <div className="text-sm">
                <div className="text-neutral-500">Features</div>
                <ul className="list-disc pl-5">
                  {item.features.map((f: any, i: number) => (
                    <li key={i}>{String(f)}</li>
                  ))}
                </ul>
              </div>
            )}
            {item.description && (
              <div className="text-sm">
                <div className="text-neutral-500">Description</div>
                {Array.isArray(item.description) ? (
                  <ul className="list-disc pl-5">
                    {item.description.map((d: any, i: number) => (
                      <li key={i}>{String(d)}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="whitespace-pre-wrap">{String(item.description)}</div>
                )}
              </div>
            )}
            {Array.isArray(item.categories) && item.categories.length > 0 && (
              <div className="text-sm">
                <div className="text-neutral-500">Categories</div>
                <div>
                  {(() => {
                    const cat = item.categories as any
                    if (Array.isArray(cat) && Array.isArray(cat[0])) {
                      return (cat[0] as any[]).join(' › ')
                    }
                    if (Array.isArray(cat)) return cat.join(' / ')
                    return String(cat)
                  })()}
                </div>
              </div>
            )}
            {item.details && (
              <div className="text-sm">
                <div className="text-neutral-500">Details</div>
                {(() => {
                  const raw: any = (item as any).details
                  let obj: any = raw
                  if (typeof raw === 'string') {
                    try { obj = JSON.parse(raw) } catch { obj = raw }
                  }
                  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
                    const entries = Object.entries(obj as Record<string, any>)
                    if (entries.length === 0) return <div className="text-neutral-500">—</div>
                    return (
                      <div className="grid grid-cols-2 gap-2">
                        {entries.slice(0, 12).map(([k, v]) => (
                          <div key={k} className="flex flex-col">
                            <div className="text-neutral-500">{k}</div>
                            <div className="break-words">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                          </div>
                        ))}
                      </div>
                    )
                  }
                  return <div className="whitespace-pre-wrap break-words">{String(raw)}</div>
                })()}
              </div>
            )}
            {item.parent_asin && (
              <div className="text-sm">
                <div className="text-neutral-500">Parent ASIN</div>
                <div className="break-all">{item.parent_asin}</div>
              </div>
            )}
            {Array.isArray(item.videos) && item.videos.length > 0 && (
              <div className="text-sm">
                <div className="text-neutral-500">Videos</div>
                <ul className="list-disc pl-5">
                  {item.videos.map((v: any, i: number) => (
                    <li key={i}>
                      {v?.title ? `${v.title} — ` : ''}
                      {v?.url ? (<a className="text-blue-600 hover:underline" href={v.url} target="_blank" rel="noreferrer">{v.url}</a>) : String(v)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(item.bought_together) && item.bought_together.length > 0 && (
              <div className="text-sm">
                <div className="text-neutral-500">Bought together</div>
                <ul className="list-disc pl-5">
                  {item.bought_together.map((b: any, i: number) => (
                    <li key={i}>{typeof b === 'string' ? b : JSON.stringify(b)}</li>
                  ))}
                </ul>
              </div>
            )}
            {item.why && (
              <div className="text-sm">
                <div className="text-neutral-500">Why selected</div>
                <div className="whitespace-pre-wrap">{item.why}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


