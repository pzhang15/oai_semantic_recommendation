import type { SearchItem } from '../lib/types'
import { fmtPrice, fmtScore } from '../lib/format'

export default function ProductCard({ item, onSelect }: { item: SearchItem, onSelect?: (item: SearchItem)=>void }) {
  const Wrapper: React.ElementType = onSelect ? 'button' : 'a'
  const wrapperProps = onSelect
    ? { onClick: ()=>onSelect(item) }
    : { href: item.product_url || '#', target: '_blank', rel: 'noreferrer' }
  return (
    <Wrapper className="text-left w-full border rounded overflow-hidden hover:shadow transition bg-white dark:bg-neutral-900" {...(wrapperProps as any)}>
      {item.image_url ? (
        <img src={item.image_url} alt={item.title} className="w-full h-48 object-cover" loading="lazy" />
      ) : (
        <div className="w-full h-48 bg-neutral-200 dark:bg-neutral-800" />
      )}
      <div className="p-3">
        <div className="text-sm font-medium line-clamp-2">{item.title}</div>
        <div className="text-xs text-neutral-500">{item.brand}</div>
        {item.average_rating != null && (
          <div className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">
            <span className="mr-1">★ {item.average_rating?.toFixed(1)}</span>
            {typeof item.rating_number === 'number' && isFinite(item.rating_number) && (
              <span>• {item.rating_number}</span>
            )}
          </div>
        )}
        <div className="flex items-center justify-between mt-2 text-sm">
          <span>{item.price != null ? fmtPrice(item.price) : 'No price available'}</span>
          <span className="text-xs px-2 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800">{fmtScore(item.score)}</span>
        </div>
        {item.why && <div title={item.why} className="text-[11px] text-neutral-500 mt-1 line-clamp-2">{item.why}</div>}
      </div>
    </Wrapper>
  )
}


