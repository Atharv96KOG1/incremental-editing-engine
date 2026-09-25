import { useRef } from 'react'

export default function Composer({ value, onChange, onSend, disabled, placeholder }) {
  const textareaRef = useRef(null)

  const handleInput = (e) => {
    onChange(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 200) + 'px'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={textareaRef}
          className="composer-input"
          rows={1}
          value={value}
          placeholder={placeholder}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button className="composer-send" onClick={onSend} disabled={disabled || !value.trim()}>
          ↑
        </button>
      </div>
      <div className="composer-hint">Describes a change to an existing file, or a new file to create.</div>
    </div>
  )
}
