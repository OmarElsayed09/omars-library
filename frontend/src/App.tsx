import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import ReactMarkdown from 'react-markdown'
import {
  ArrowUp,
  Check,
  ChevronDown,
  Download,
  FilePlus2,
  FileText,
  Globe2,
  History,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  ThumbsDown,
  ThumbsUp,
  Upload,
  X,
} from 'lucide-react'
import './App.css'

type Language = 'en' | 'ar'
type ReplyLanguage = 'auto' | Language
type Panel = 'history' | 'library' | null
type Role = 'user' | 'assistant'

interface Chat {
  id: string
  title: string
  updated_at: string
}

interface Source {
  source: string
  content: string
  chunk: number
  page?: number | null
}

interface Message {
  id: number | string
  role: Role
  content: string
  sources: Source[]
  created_at?: string
}

interface DocumentInfo {
  source: string
  chunks: number
}

interface Library {
  documents: DocumentInfo[]
  chunks: number
}

interface Status {
  ollama_connected: boolean
  chat_model: string
  embedding_model: string
  error: string
}

interface RetrievalTrace {
  route?: string
  original_query?: string
  retrieval_query?: string
  dense?: Array<{ source?: string; chunk?: number; distance?: number }>
  bm25?: Array<{ source?: string; chunk?: number; score?: number }>
  fused?: Array<{ source?: string; chunk?: number; rrf_score?: number }>
  reranked?: boolean
  reranker_available?: boolean
  rerank_latency_ms?: number
  selected?: Source[]
}

const translations = {
  en: {
    brand: 'Omar’s Library',
    newChat: 'New conversation',
    history: 'History',
    searchHistory: 'Find a conversation',
    conversations: 'Conversations',
    library: 'Your library',
    addFiles: 'Import documents',
    sync: 'Refresh index',
    exportChat: 'Export this conversation',
    backup: 'Download backup',
    restore: 'Restore backup',
    uploadProgress: 'Uploading',
    indexing: 'Preparing the search index…',
    indexed: 'Import complete. Indexed files are ready for questions.',
    retryUpload: 'Retry import',
    preparing: 'Preparing the answer…',
    stopReply: 'Stop response',
    replyStopped: 'Response stopped. You can resend your question.',
    restoring: 'Restoring backup',
    restoreSummary: (chats: number, documents: number) =>
      `Backup merged: ${chats} conversations and ${documents} documents added.`,
    you: 'You',
    emptyChats: 'Past conversations will appear here.',
    emptyDocs: 'Your library is waiting for its first document.',
    welcome: 'What would you like to explore?',
    libraryHeading: 'Your documents',
    welcomeBody: 'Choose a file to ask about it and open the sources cited in the answer.',
    firstFile: 'Upload your first file',
    excerpt: 'View excerpt',
    eyebrow: 'YOUR PRIVATE KNOWLEDGE STUDIO',
    suggestions: ['Find the key points', 'Summarize a document', 'Start a conversation'],
    starterHeading: 'Try a question',
    placeholder: 'Write a message…',
    documentPlaceholder: 'Ask about this document…',
    scope: 'Search scope',
    automatic: 'All library (automatic)',
    hint: 'Enter to send · Shift + Enter for a new line',
    send: 'Send message',
    privacy: 'Private by design',
    local: 'LOCAL',
    interface: 'Interface',
    response: 'Response language',
    auto: 'Follow the conversation',
    english: 'English',
    arabic: 'العربية',
    sources: 'Sources',
    copyCode: 'Copy code',
    copied: 'Copied',
    copyFailed: 'Select the code to copy it',
    page: 'Page',
    chunks: 'indexed passages',
    documents: 'documents',
    remove: 'Remove document',
    confirmChat: 'Delete this conversation?',
    confirmDelete: 'Delete',
    cancelDelete: 'Cancel',
    confirmFile: 'Remove this file and its indexed content?',
    thinking: 'Thinking',
    searching: 'Looking through your library',
    ready: 'All systems ready',
    offline: 'Local service unavailable',
    loadError: 'Could not load the workspace.',
    chatError: 'Could not start a conversation.',
    messageError: 'Could not complete that request.',
    syncError: 'Could not refresh the document index.',
    deleteError: 'Could not remove that item.',
    uploadError: 'Could not import that document.',
    noResults: 'No matching conversations.',
    close: 'Close',
    settings: 'Preferences',
    developerDebug: 'Developer retrieval details',
    helpful: 'Helpful',
    notHelpful: 'Not helpful',
    feedbackSaved: 'Thanks — saved locally.',
    dismiss: 'Dismiss',
  },
  ar: {
    brand: 'مكتبة عمر',
    newChat: 'محادثة جديدة',
    history: 'السجل',
    searchHistory: 'ابحث عن محادثة',
    conversations: 'المحادثات',
    library: 'مكتبتك',
    addFiles: 'استيراد مستندات',
    sync: 'تحديث الفهرس',
    exportChat: 'تصدير هذه المحادثة',
    backup: 'تنزيل نسخة احتياطية',
    restore: 'استعادة نسخة احتياطية',
    uploadProgress: 'جارٍ الرفع',
    indexing: 'جارٍ تجهيز فهرس البحث…',
    indexed: 'اكتمل الاستيراد. الملفات المفهرسة جاهزة للأسئلة.',
    retryUpload: 'إعادة محاولة الاستيراد',
    preparing: 'بجهّز الإجابة…',
    stopReply: 'إيقاف الرد',
    replyStopped: 'تم إيقاف عرض الرد. تقدر تبعت السؤال تاني.',
    restoring: 'جارٍ استعادة النسخة الاحتياطية',
    restoreSummary: (chats: number, documents: number) =>
      `تم دمج النسخة الاحتياطية: أُضيفت ${chats} محادثات و${documents} مستندات.`,
    you: 'أنت',
    emptyChats: 'ستظهر محادثاتك السابقة هنا.',
    emptyDocs: 'مكتبتك جاهزة لاستقبال أول مستند.',
    welcome: 'حابب تسأل عن إيه؟',
    libraryHeading: 'مستنداتك',
    welcomeBody: 'اختار ملف واسأل عن محتواه، وافتح المصادر المذكورة في الإجابة.',
    firstFile: 'ارفع أول ملف',
    excerpt: 'عرض المقطع',
    eyebrow: 'مساحة معرفية خاصة بك',
    suggestions: ['استخرج أهم النقاط', 'لخّص مستندًا', 'ابدأ محادثة'],
    starterHeading: 'جرّب سؤال',
    placeholder: 'اكتب رسالة…',
    documentPlaceholder: 'اسأل عن الملف ده…',
    scope: 'نطاق البحث',
    automatic: 'كل المكتبة (تلقائي)',
    hint: 'Enter للإرسال · Shift + Enter لسطر جديد',
    send: 'إرسال',
    privacy: 'خصوصيتك أولوية',
    local: 'محلي',
    interface: 'الواجهة',
    response: 'لغة الرد',
    auto: 'حسب لغة المحادثة',
    english: 'English',
    arabic: 'العربية',
    sources: 'المصادر',
    copyCode: 'نسخ الكود',
    copied: 'تم النسخ',
    copyFailed: 'حدّد الكود لنسخه',
    page: 'صفحة',
    chunks: 'مقاطع مفهرسة',
    documents: 'مستندات',
    remove: 'حذف المستند',
    confirmChat: 'هل تريد حذف هذه المحادثة؟',
    confirmDelete: 'حذف',
    cancelDelete: 'إلغاء',
    confirmFile: 'هل تريد حذف الملف ومحتواه المفهرس؟',
    thinking: 'جارٍ التفكير',
    searching: 'جارٍ البحث في مكتبتك',
    ready: 'كل شيء جاهز',
    offline: 'الخدمة المحلية غير متاحة',
    loadError: 'تعذر تحميل المساعد المحلي.',
    chatError: 'تعذر بدء المحادثة.',
    messageError: 'تعذرت إكمال الإجابة.',
    syncError: 'تعذر تحديث فهرس المستندات.',
    deleteError: 'تعذر حذف العنصر.',
    uploadError: 'تعذر استيراد المستند.',
    noResults: 'لا توجد محادثات مطابقة.',
    close: 'إغلاق',
    settings: 'التفضيلات',
    developerDebug: 'تفاصيل البحث للمطور',
    helpful: 'مفيد',
    notHelpful: 'غير مفيد',
    feedbackSaved: 'شكرًا — تم الحفظ محليًا.',
    dismiss: 'إخفاء',
  },
} as const

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail || `Request failed (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return await response.json() as T
}

function sourceId(messageId: string | number, index: number) {
  return `source-${messageId}-${index}`
}

function pdfPageUrl(source: Source) {
  return `/api/documents/pdf?source=${encodeURIComponent(source.source)}#page=${source.page}`
}

function linkCitations(content: string, sources: Source[], messageId: string | number) {
  return content.split(/(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g)
    .map((part, index) => index % 2 ? part : linkProseCitations(part, sources, messageId))
    .join('')
}

function linkProseCitations(content: string, sources: Source[], messageId: string | number) {
  content = content.replace(/\[(\d+)\](?!\()/g, (original, value: string) => {
    const index = Number(value) - 1
    return sources[index] ? `[${value}](#${sourceId(messageId, index)})` : original
  })
  return sources.reduce((answer, source, index) => {
    const escapedSource = source.source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const target = `#${sourceId(messageId, index)}`
    let linked = answer.replace(
      new RegExp(`\\[${escapedSource}\\]`, 'g'),
      `[${source.source}](${target})`,
    )
    if (source.page) {
      const citation = `${source.source}, page ${source.page}`
      const escapedCitation = citation.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      linked = linked.replace(
        new RegExp(`\\[${escapedCitation}\\]`, 'g'),
        `[${citation}](${target})`,
      )
      linked = linked.replace(
        new RegExp(`\\[${escapedCitation}, chunk ${source.chunk}\\]`, 'g'),
        `[${index + 1}](${target})`,
      )
    }
    return linked
  }, content)
}

function App() {
  const [language, setLanguage] = useState<Language>(
    () => (localStorage.getItem('assistant-language') as Language) || 'en',
  )
  const [replyLanguage, setReplyLanguage] = useState<ReplyLanguage>(() => {
    const saved = localStorage.getItem('reply-language')
    return saved === 'ar' || saved === 'en' ? saved : 'auto'
  })
  const [chats, setChats] = useState<Chat[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedSource, setSelectedSource] = useState('')
  const [chunkCount, setChunkCount] = useState(0)
  const [status, setStatus] = useState<Status | null>(null)
  const [retrievalTrace, setRetrievalTrace] = useState<RetrievalTrace | null>(null)
  const [developerDebug, setDeveloperDebug] = useState(
    () => localStorage.getItem('developer-debug') === 'true',
  )
  const [historyQuery, setHistoryQuery] = useState('')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [indexing, setIndexing] = useState(false)
  const [retryFiles, setRetryFiles] = useState<File[]>([])
  const [preparing, setPreparing] = useState(false)
  const activeRequest = useRef<AbortController | null>(null)
  const [syncing, setSyncing] = useState(false)
  const [restoring, setRestoring] = useState(false)
  const [removingDocument, setRemovingDocument] = useState('')
  const [panel, setPanel] = useState<Panel>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const backupInput = useRef<HTMLInputElement>(null)
  const panelElement = useRef<HTMLElement>(null)
  const chatEnd = useRef<HTMLDivElement>(null)
  const composerInput = useRef<HTMLTextAreaElement>(null)
  const [confirmChatId, setConfirmChatId] = useState<string | null>(null)
  const [deletingChat, setDeletingChat] = useState(false)

  useEffect(() => {
    const input = composerInput.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 155)}px`
  }, [draft])
  const t = translations[language]
  const rtl = language === 'ar'
  const libraryBusy = uploading || syncing || restoring || Boolean(removingDocument)
  const filteredChats = chats.filter((chat) =>
    chat.title.toLocaleLowerCase().includes(historyQuery.toLocaleLowerCase()),
  )

  const updateLibrary = useCallback((library: Library) => {
    setDocuments(library.documents)
    setSelectedSource((current) => library.documents.some((doc) => doc.source === current) ? current : '')
    setChunkCount(library.chunks)
  }, [])

  const refreshChats = useCallback(async () => {
    const result = await api<Chat[]>('/chats')
    setChats(result)
    return result
  }, [])

  const openChat = useCallback(async (id: string) => {
    if (activeRequest.current) return
    const history = await api<Message[]>(`/chats/${id}/messages`)
    setActiveChatId(id)
    setSelectedSource('')
    setMessages(history)
  }, [])

  const createChat = useCallback(async () => {
    if (activeRequest.current) return
    try {
      const chat = await api<Chat>('/chats', { method: 'POST' })
      setChats((current) => [chat, ...current])
      setActiveChatId(chat.id)
      setSelectedSource('')
      setMessages([])
      setPanel(null)
      setError('')
      setNotice('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.chatError)
    }
  }, [setError, setNotice, t.chatError])

  useEffect(() => {
    let mounted = true
    async function initialize() {
      try {
        const [chatList, library, health] = await Promise.all([
          api<Chat[]>('/chats'),
          api<Library>('/documents'),
          api<Status>('/status'),
        ])
        if (!mounted) return
        updateLibrary(library)
        setStatus(health)
        if (chatList.length) {
          setChats(chatList)
          setActiveChatId(chatList[0].id)
          setMessages(await api<Message[]>(`/chats/${chatList[0].id}/messages`))
        } else {
          const first = await api<Chat>('/chats', { method: 'POST' })
          if (!mounted) return
          setChats([first])
          setActiveChatId(first.id)
        }
      } catch (reason) {
        if (mounted) setError(reason instanceof Error ? reason.message : t.loadError)
      }
    }
    void initialize()
    return () => { mounted = false }
  }, [t.loadError, updateLibrary])

  useEffect(() => {
    const stream = chatEnd.current?.closest('.message-stream')
    if (stream) stream.scrollTop = stream.scrollHeight
  }, [messages, busy])

  useEffect(() => {
    const handleShortcut = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        void createChat()
      } else if (event.key === 'Escape') {
        setPanel(null)
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [createChat])

  useEffect(() => {
    if (!panel || !panelElement.current) return
    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null
    const focusable = () => Array.from(
      panelElement.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]):not([type="file"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) || [],
    ).filter((element) => element.getClientRects().length > 0)
    focusable()[0]?.focus()

    const trapFocus = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Tab') return
      const elements = focusable()
      if (!elements.length) {
        event.preventDefault()
        return
      }
      const first = elements[0]
      const last = elements[elements.length - 1]
      if (event.shiftKey && (document.activeElement === first || !panelElement.current?.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !panelElement.current?.contains(document.activeElement))) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', trapFocus)
    return () => {
      document.removeEventListener('keydown', trapFocus)
      previousFocus?.focus()
    }
  }, [panel])

  const changeLanguage = (next: Language) => {
    setLanguage(next)
    localStorage.setItem('assistant-language', next)
    document.documentElement.lang = next
    document.documentElement.dir = next === 'ar' ? 'rtl' : 'ltr'
  }

  const syncLibrary = async () => {
    if (libraryBusy) return
    setSyncing(true)
    setError('')
    setNotice('')
    try {
      updateLibrary(await api<Library>('/documents/sync', { method: 'POST' }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.syncError)
    } finally {
      setSyncing(false)
    }
  }

  const uploadFiles = async (files: FileList | File[] | null) => {
    if (!files?.length || libraryBusy) return
    setUploading(true)
    setIndexing(false)
    setRetryFiles([])
    setUploadProgress(0)
    setError('')
    setNotice('')
    try {
      const form = new FormData()
      Array.from(files).forEach((file) => form.append('files', file))
      const library = await new Promise<Library>((resolve, reject) => {
        const request = new XMLHttpRequest()
        request.open('POST', '/api/documents/upload')
        request.upload.onload = () => setIndexing(true)
        request.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            setUploadProgress(Math.round((event.loaded / event.total) * 100))
          }
        }
        request.onerror = () => reject(new Error('Network error while uploading documents.'))
        request.onload = () => {
          if (request.status < 200 || request.status >= 300) {
            let detail = ''
            try {
              const body = JSON.parse(request.responseText)
              if (typeof body?.detail === 'string') detail = body.detail
            } catch { /* Use the HTTP status for non-JSON errors. */ }
            reject(new Error(detail || `${t.uploadError} (${request.status})`))
            return
          }
          let body: { detail?: string } & Partial<Library> = {}
          try {
            const parsed: unknown = JSON.parse(request.responseText)
            if (
              !parsed
              || typeof parsed !== 'object'
              || !('documents' in parsed)
              || !Array.isArray(parsed.documents)
              || !('chunks' in parsed)
              || typeof parsed.chunks !== 'number'
            ) {
              reject(new Error(`Upload returned an invalid response (${request.status}).`))
              return
            }
            body = parsed as Library
          } catch {
            reject(new Error(`Upload failed (${request.status}).`))
            return
          }
          resolve(body as Library)
        }
        request.send(form)
      })
      updateLibrary(library)
      setNotice(t.indexed)
    } catch (reason) {
      setRetryFiles(Array.from(files))
      setError(reason instanceof Error ? reason.message : t.uploadError)
    } finally {
      setUploading(false)
      setIndexing(false)
      setUploadProgress(0)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const exportConversation = () => {
    const activeChat = chats.find((chat) => chat.id === activeChatId)
    if (!activeChat || !messages.length) return
    const markdown = messages.map((message) => {
      const speaker = message.role === 'user' ? t.you : t.brand
      const sources = message.sources.length
        ? `\n\nSources:\n${message.sources.map((source) => `- ${source.source}${source.page ? `, ${t.page} ${source.page}` : ''}`).join('\n')}`
        : ''
      return `## ${speaker}\n\n${message.content}${sources}`
    }).join('\n\n---\n\n')
    const blob = new Blob([`# ${activeChat.title}\n\n${markdown}\n`], {
      type: 'text/markdown;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    const safeTitle = Array.from(activeChat.title, (character) => (
      character.charCodeAt(0) < 32 || '<>:"/\\|?*'.includes(character) ? '_' : character
    )).join('')
    link.download = `${safeTitle || 'conversation'}.md`
    link.click()
    URL.revokeObjectURL(url)
  }

  const downloadBackup = async () => {
    if (libraryBusy) return
    setError('')
    setNotice('')
    try {
      const response = await fetch('/api/backup')
      if (!response.ok) {
        const body = await response.json().catch(() => null) as { detail?: string } | null
        throw new Error(body?.detail || `Backup failed (${response.status}).`)
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'omars-library-backup.zip'
      link.click()
      URL.revokeObjectURL(url)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.loadError)
    }
  }

  const restoreBackup = async (file: File | undefined) => {
    if (!file || libraryBusy) return
    setRestoring(true)
    setError('')
    setNotice('')
    try {
      const form = new FormData()
      form.append('file', file)
      const result = await api<{ documents_added: number; chats_added: number }>(
        '/backup/restore',
        { method: 'POST', body: form },
      )
      const [library, chatList] = await Promise.all([
        api<Library>('/documents'),
        refreshChats(),
      ])
      updateLibrary(library)
      setChats(chatList)
      if (activeChatId && chatList.some((chat) => chat.id === activeChatId)) {
        setMessages(await api<Message[]>(`/chats/${activeChatId}/messages`))
      } else if (chatList.length) {
        await openChat(chatList[0].id)
      }
      setNotice(t.restoreSummary(result.chats_added, result.documents_added))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.loadError)
    } finally {
      setRestoring(false)
      if (backupInput.current) backupInput.current.value = ''
    }
  }

  const removeDocument = async (source: string) => {
    if (libraryBusy) return
    if (!window.confirm(`${t.confirmFile}\n\n${source}`)) return
    setRemovingDocument(source)
    try {
      updateLibrary(await api<Library>(
        `/documents?source=${encodeURIComponent(source)}`,
        { method: 'DELETE' },
      ))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.deleteError)
    } finally {
      setRemovingDocument('')
    }
  }

  const deleteChat = async (chat: Chat) => {
    if (activeRequest.current || deletingChat) return
    setDeletingChat(true)
    setError('')
    try {
      await api<void>(`/chats/${chat.id}`, { method: 'DELETE' })
      const remaining = chats.filter((item) => item.id !== chat.id)
      setChats((current) => current.filter((item) => item.id !== chat.id))
      setConfirmChatId(null)
      if (activeChatId === chat.id) {
        setActiveChatId(null)
        setMessages([])
        if (remaining.length) await openChat(remaining[0].id)
        else await createChat()
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t.deleteError)
    } finally {
      setDeletingChat(false)
    }
  }

  const chatDeleteControl = (chat: Chat) => confirmChatId === chat.id ? (
    <div className="chat-delete-confirm" role="group" aria-label={t.confirmChat}>
      <button type="button" disabled={busy || deletingChat} onClick={() => void deleteChat(chat)}>{t.confirmDelete}</button>
      <button type="button" disabled={deletingChat} onClick={() => setConfirmChatId(null)}>{t.cancelDelete}</button>
    </div>
  ) : (
    <button type="button" className="history-delete" disabled={busy || deletingChat}
      onClick={() => setConfirmChatId(chat.id)} aria-label={`${t.confirmDelete}: ${chat.title}`} title={t.confirmDelete}>
      <X size={14} />
    </button>
  )

  const sendMessage = async (value = draft) => {
    const content = value.trim()
    if (!content || busy || deletingChat || activeRequest.current) return
    const controller = new AbortController()
    activeRequest.current = controller
    setPreparing(false)
    const pendingId = `pending-${Date.now()}`
    const assistantId = `stream-${Date.now()}`
    setDraft('')
    setBusy(true)
    setError('')
    setNotice('')
    setMessages((current) => [
      ...current,
      { id: pendingId, role: 'user', content, sources: [] },
      { id: assistantId, role: 'assistant', content: '', sources: [] },
    ])
    let responseCompleted = false
    let completedMessageId: number | null = null
    try {
      let chatId = activeChatId
      if (!chatId) {
        const chat = await api<Chat>('/chats', { method: 'POST', signal: controller.signal })
        chatId = chat.id
        setChats((current) => [chat, ...current])
        setActiveChatId(chatId)
      }
      const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: content, language: replyLanguage, source: selectedSource || null }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null) as { detail?: string } | null
        throw new Error(body?.detail || `Request failed (${response.status})`)
      }
      if (!response.body) throw new Error('The streaming response is unavailable.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finished = false
      const handleEvent = (block: string) => {
        const data = block.split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('\n')
        if (!data) return
        const event = JSON.parse(data) as
          | { type: 'sources'; sources: Source[]; trace?: RetrievalTrace }
          | { type: 'delta'; content: string }
          | { type: 'done'; message_id?: number }
          | { type: 'error'; detail: string }
        if (event.type === 'sources') {
          setPreparing(true)
          setRetrievalTrace(event.trace || null)
          setMessages((current) => current.map((message) =>
            message.id === assistantId ? { ...message, sources: event.sources } : message,
          ))
        } else if (event.type === 'delta') {
          setMessages((current) => current.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + event.content }
              : message,
          ))
        } else if (event.type === 'done') {
          finished = true
          completedMessageId = event.message_id ?? null
        } else {
          throw new Error(event.detail)
        }
      }
      while (true) {
        const { value: chunk, done } = await reader.read()
        buffer += decoder.decode(chunk, { stream: !done })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() || ''
        blocks.forEach(handleEvent)
        if (done) break
      }
      if (buffer.trim()) handleEvent(buffer)
      if (!finished) throw new Error(t.messageError)
      responseCompleted = true
      setMessages((current) => current
        .map((message) => message.id === assistantId
          ? { ...message, id: completedMessageId ?? `answer-${Date.now()}` }
          : message))
      try {
        await refreshChats()
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : t.loadError)
      }
    } catch (reason) {
      if (!responseCompleted) {
        setDraft((current) => current || content)
        setMessages((current) => current.filter(
          (message) => message.id !== pendingId && message.id !== assistantId,
        ))
      }
      if (controller.signal.aborted) {
        setDraft(content)
        setNotice(t.replyStopped)
      } else {
        setError(reason instanceof Error ? reason.message : t.messageError)
      }
    } finally {
      activeRequest.current = null
      setBusy(false)
    }
  }

  const handleComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      void sendMessage()
    }
  }

  return (
    <div className={`app-shell ${rtl ? 'rtl' : ''}`} dir={rtl ? 'rtl' : 'ltr'}>
      <input
        ref={fileInput}
        className="hidden-input"
        type="file"
        accept=".txt,.md,.markdown,.pdf,.docx,.csv,.py,.js,.jsx,.ts,.tsx,.css,.java,.go,.rs,.cpp,.cs"
        multiple
        onChange={(event) => void uploadFiles(event.target.files)}
      />
      {panel && (
        <button
          className="panel-backdrop"
          onClick={() => setPanel(null)}
          aria-label={t.close}
        />
      )}

      <header className="app-header" aria-hidden={panel !== null} inert={panel !== null}>
        <button className="brand-lockup" onClick={() => setPanel(null)} aria-label={t.brand}>
          <span className="brand-symbol"><FileText size={16} /></span>
          <span className="brand-name">{t.brand}</span>
          <span className="brand-divider" />
          <span className="brand-descriptor">{t.local}</span>
        </button>

        <nav className="header-controls" aria-label={t.brand}>
          <button className="header-new-chat" onClick={() => void createChat()} aria-label={t.newChat} title={t.newChat}>
            <Plus size={16} /><span>{t.newChat}</span>
          </button>
          <div className="connection-indicator">
            <span className={`connection-light ${status?.ollama_connected ? 'is-ready' : ''}`} />
            <span>{status?.ollama_connected ? t.ready : t.offline}</span>
          </div>
          <button
            className={`header-tool ${panel === 'history' ? 'header-tool-active' : ''}`}
            onClick={() => setPanel(panel === 'history' ? null : 'history')}
            aria-label={t.history}
            title={t.history}
          >
            <History size={16} /><span>{t.history}</span>
          </button>
          <button
            className={`header-tool ${panel === 'library' ? 'header-tool-active' : ''}`}
            onClick={() => setPanel(panel === 'library' ? null : 'library')}
            aria-label={t.library}
            title={t.library}
          >
            <FileText size={16} /><span>{t.library}</span>
            {documents.length > 0 && <span className="header-count">{documents.length}</span>}
          </button>
          <button
            className="language-toggle"
            onClick={() => changeLanguage(rtl ? 'en' : 'ar')}
            aria-label={`${t.interface}: ${rtl ? t.english : t.arabic}`}
            title={t.interface}
          >
            <Globe2 size={15} /><span>{rtl ? 'EN' : 'عربي'}</span>
          </button>
          <section className="sidebar-history" aria-label={t.conversations}>
            <h2>{t.conversations}</h2>
            {chats.slice(0, 8).map((chat) => (
              <div className="history-item" key={chat.id}>
                <button title={chat.title} aria-current={chat.id === activeChatId ? 'true' : undefined}
                  disabled={busy || deletingChat} onClick={() => void openChat(chat.id).catch(() => setError(t.loadError))}>
                  <span className="history-item-title">{chat.title}</span>
                </button>
                {chatDeleteControl(chat)}
              </div>
            ))}
          </section>
        </nav>
      </header>

      {panel && (
        <aside ref={panelElement} className={`resource-panel ${rtl ? 'resource-panel-rtl' : ''}`} aria-label={panel === 'history' ? t.history : t.library} aria-modal="true" role="dialog" aria-busy={libraryBusy}>
          <div className="panel-heading">
            <div>
              <span className="panel-kicker">{t.brand} / {t.local}</span>
              <h2>{panel === 'history' ? t.conversations : t.library}</h2>
            </div>
            <button className="panel-close" onClick={() => setPanel(null)} aria-label={t.close}><X size={18} /></button>
          </div>

          {panel === 'history' ? (
            <>
              <button className="panel-new-chat" onClick={() => void createChat()}>
                <Plus size={16} />{t.newChat}
              </button>
              <button className="panel-secondary-action export-chat-button" onClick={exportConversation} disabled={!messages.length || busy}>
                <Download size={15} />{t.exportChat}
              </button>
              <label className="panel-search">
                <Search size={16} />
                <input
                  value={historyQuery}
                  onChange={(event) => setHistoryQuery(event.target.value)}
                  placeholder={t.searchHistory}
                  aria-label={t.searchHistory}
                />
              </label>
              <div className="panel-list">
                {filteredChats.map((chat) => (
                  <div className={`history-item ${chat.id === activeChatId ? 'history-item-active' : ''}`} key={chat.id}>
                    <button
                      onClick={() => {
                        void openChat(chat.id)
                        setPanel(null)
                      }}
                      title={chat.title}
                    >
                      <span className="history-item-icon"><History size={15} /></span>
                      <span className="history-item-title">{chat.title}</span>
                    </button>
                    {chatDeleteControl(chat)}
                  </div>
                ))}
                {!filteredChats.length && <p className="panel-empty">{t.noResults}</p>}
              </div>
            </>
          ) : (
            <>
              <div className="library-intro">
                <span className="library-orbit"><FileText size={19} /></span>
                <div>
                  <strong>{documents.length} {t.documents}</strong>
                  <span>{chunkCount} {t.chunks}</span>
                </div>
              </div>
              <div className="panel-actions">
                <button className="panel-primary-action" onClick={() => fileInput.current?.click()} disabled={libraryBusy}>
                  {uploading ? <LoaderCircle className="spin" size={16} /> : <FilePlus2 size={16} />}
                  {t.addFiles}
                </button>
                <button className="panel-secondary-action" onClick={() => void syncLibrary()} disabled={libraryBusy}>
                  {syncing ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={15} />}
                  {t.sync}
                </button>
              </div>
              {uploading && (
                <div className="upload-progress" role="status" aria-live="polite">
                  <span>{indexing ? t.indexing : `${t.uploadProgress}: ${uploadProgress}%`}</span>
                  <progress value={indexing ? undefined : uploadProgress} max="100" aria-label={indexing ? t.indexing : t.uploadProgress} />
                </div>
              )}
              <div className="backup-actions">
                <button className="panel-secondary-action" onClick={() => void downloadBackup()} disabled={libraryBusy}>
                  <Download size={15} />{t.backup}
                </button>
                <button className="panel-secondary-action" onClick={() => backupInput.current?.click()} disabled={libraryBusy}>
                  {restoring ? <LoaderCircle className="spin" size={15} /> : <Upload size={15} />}
                  {restoring ? t.restoring : t.restore}
                </button>
                <input
                  ref={backupInput}
                  className="hidden-input"
                  type="file"
                  accept=".zip,application/zip"
                  onChange={(event) => void restoreBackup(event.target.files?.[0])}
                />
              </div>
              <p className="panel-section-label">{t.library}</p>
              <div className="panel-list document-panel-list">
                {documents.map((document) => (
                  <div className="library-document" key={document.source}>
                    <span className="document-type"><FileText size={16} /></span>
                    <span className="library-document-copy">
                      <strong title={document.source}>{document.source}</strong>
                      <small>{document.chunks} {t.chunks}</small>
                    </span>
                    <button
                      className="document-remove"
                      onClick={() => void removeDocument(document.source)}
                      title={t.remove}
                      aria-label={`${t.remove}: ${document.source}`}
                      disabled={libraryBusy}
                    >
                      <X size={15} />
                    </button>
                  </div>
                ))}
                {!documents.length && <p className="panel-empty">{t.emptyDocs}</p>}
              </div>
              <div className="panel-preferences">
                <div className="preference-heading"><Settings2 size={15} /><span>{t.settings}</span></div>
                <label className="preference-row">
                  <span><Globe2 size={15} />{t.interface}</span>
                  <select value={language} onChange={(event) => changeLanguage(event.target.value as Language)}>
                    <option value="en">{t.english}</option><option value="ar">{t.arabic}</option>
                  </select>
                </label>
                <label className="preference-row">
                  <span><span className="type-glyph">Aa</span>{t.response}</span>
                  <select value={replyLanguage} onChange={(event) => {
                    const value = event.target.value as ReplyLanguage
                    setReplyLanguage(value)
                    localStorage.setItem('reply-language', value)
                  }}>
                    <option value="auto">{t.auto}</option><option value="en">{t.english}</option><option value="ar">{t.arabic}</option>
                  </select>
                </label>
                <label className="preference-row">
                  <span><Search size={15} />{t.developerDebug}</span>
                  <input
                    type="checkbox"
                    checked={developerDebug}
                    onChange={(event) => {
                      setDeveloperDebug(event.target.checked)
                      localStorage.setItem('developer-debug', String(event.target.checked))
                    }}
                  />
                </label>
              </div>
            </>
          )}
          {notice && (
            <div className="success-banner" role="status" aria-live="polite">
              <span>{notice}</span>
              <button onClick={() => setNotice('')} aria-label={t.dismiss}><X size={15} /></button>
            </div>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              {retryFiles.length > 0 && <button type="button" disabled={libraryBusy} onClick={() => void uploadFiles(retryFiles)}>{t.retryUpload}</button>}
              <button onClick={() => setError('')} aria-label={t.dismiss}><X size={15} /></button>
            </div>
          )}
          <div className="panel-footnote"><ShieldCheck size={14} /><span>{t.privacy}</span></div>
        </aside>
      )}

      <main className="main-panel" aria-hidden={panel !== null} inert={panel !== null}>
        <header className="workspace-bar">
          <div className="workspace-title">
            <span>{t.brand}</span>
            <strong dir="auto">{chats.find((chat) => chat.id === activeChatId)?.title || t.newChat}</strong>
          </div>
          <span className="workspace-privacy"><ShieldCheck size={16} aria-hidden="true" />{t.privacy}</span>
        </header>
        <section className={`conversation-stage ${messages.length ? 'has-messages' : ''}`}>
          <div className="message-stream">
            {messages.length > 0 && (
              <div className="message-list">
                {messages.map((message, index) => (
                  <MessageBubble key={message.id} message={message} index={index} labels={t} chatId={activeChatId} />
                ))}
                {busy && (
                  <div className="assistant-row">
                    <div className="assistant-sigil"><Search size={14} /></div>
                    <div className="thinking-indicator" role="status" aria-live="polite">
                      <span className="thinking-dots"><i /><i /><i /></span>
                      {preparing ? t.preparing : documents.length ? t.searching : t.thinking}
                    </div>
                  </div>
                )}
                <div ref={chatEnd} />
              </div>
            )}
          </div>

          {developerDebug && retrievalTrace && (
            <details className="retrieval-debug" open>
              <summary>{t.developerDebug}</summary>
              <div className="retrieval-debug-grid">
                <div><strong>Route</strong><span>{retrievalTrace.route || '-'}</span></div>
                <div><strong>Query</strong><span>{retrievalTrace.retrieval_query || '-'}</span></div>
                <div><strong>Dense</strong><span>{retrievalTrace.dense?.length || 0}</span></div>
                <div><strong>BM25</strong><span>{retrievalTrace.bm25?.length || 0}</span></div>
                <div><strong>Fused</strong><span>{retrievalTrace.fused?.length || 0}</span></div>
                <div><strong>Reranked</strong><span>{retrievalTrace.reranked ? 'yes' : 'no'} ({retrievalTrace.rerank_latency_ms || 0} ms)</span></div>
              </div>
            </details>
          )}

          {notice && (
            <div className="success-banner" role="status" aria-live="polite">
              <span>{notice}</span>
              <button onClick={() => setNotice('')} aria-label={t.dismiss}><X size={15} /></button>
            </div>
          )}
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              {retryFiles.length > 0 && <button type="button" disabled={libraryBusy} onClick={() => void uploadFiles(retryFiles)}>{t.retryUpload}</button>}
              <button onClick={() => setError('')} aria-label={t.dismiss}><X size={15} /></button>
            </div>
          )}

          <div className="composer-wrap">
            {!messages.length && (
              <div className="chat-welcome">
                <h1>{t.welcome}</h1>
                <p>{t.welcomeBody}</p>
              </div>
            )}
            {documents.length > 0 && (
              <details className="composer-scope"
                onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) event.currentTarget.open = false
                }}
                onKeyDown={(event) => {
                  if (event.key !== 'Escape' || !event.currentTarget.open) return
                  event.stopPropagation()
                  event.currentTarget.open = false
                  event.currentTarget.querySelector('summary')?.focus()
                }}>
                <summary aria-label={`${t.scope}: ${selectedSource || t.automatic}`}
                  title={selectedSource || t.automatic} aria-disabled={busy}
                  onClick={(event) => { if (busy) event.preventDefault() }}>
                  <FileText size={15} aria-hidden="true" />
                  <span dir="auto">{selectedSource || t.automatic}</span>
                  <ChevronDown size={14} aria-hidden="true" />
                </summary>
                <div className="scope-options" role="group" aria-label={t.scope}>
                  {['', ...documents.map(({ source }) => source)].map((source) => (
                    <button key={source} type="button" disabled={busy} aria-pressed={source === selectedSource}
                      onClick={(event) => {
                        setSelectedSource(source)
                        const scope = event.currentTarget.closest('details')
                        if (scope) scope.open = false
                        scope?.querySelector('summary')?.focus()
                      }}>
                      <span dir="auto">{source || t.automatic}</span>
                      {source === selectedSource && <Check size={15} aria-hidden="true" />}
                    </button>
                  ))}
                </div>
              </details>
            )}
            <form
              className="composer"
              onSubmit={(event: FormEvent) => {
                event.preventDefault()
                void sendMessage()
              }}
            >
              <textarea
                ref={composerInput}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKey}
                placeholder={selectedSource ? t.documentPlaceholder : t.placeholder}
                rows={1}
                aria-label={selectedSource ? t.documentPlaceholder : t.placeholder}
              />
              <div className="composer-bottom">
                <div className="composer-context">
                  <button className="composer-attach" type="button" disabled={libraryBusy}
                    onClick={() => fileInput.current?.click()} aria-label={t.addFiles} title={t.addFiles}>
                    {uploading ? <LoaderCircle className="spin" size={18} /> : <Plus size={21} />}
                  </button>
                  {uploading && <span role="status">{indexing ? t.indexing : `${t.uploadProgress} ${uploadProgress}%`}</span>}
                </div>
                <div className="composer-controls">
                  {busy ? (
                    <button className="send-button" type="button" aria-label={t.stopReply} title={t.stopReply}
                      onClick={() => activeRequest.current?.abort()}>
                      <Square size={13} fill="currentColor" />
                    </button>
                  ) : <button
                    className="send-button"
                    type="submit"
                    disabled={!draft.trim() || deletingChat}
                    aria-label={t.send}
                    title={t.send}
                  >
                    <ArrowUp size={19} />
                  </button>}
                </div>
              </div>
            </form>
          </div>
        </section>
      </main>
    </div>
  )
}

function CodeBlock({ children, labels }: { children: ReactNode, labels: (typeof translations)[Language] }) {
  const pre = useRef<HTMLPreElement>(null)
  const [feedback, setFeedback] = useState('')
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(pre.current?.textContent ?? '')
      setFeedback(labels.copied)
    } catch {
      setFeedback(labels.copyFailed)
    }
  }
  return <div className="answer-code">
    <button type="button" onClick={copy}>{labels.copyCode}</button>
    <span role="status">{feedback}</span>
    <pre dir="ltr" ref={pre}>{children}</pre>
  </div>
}

function MessageBubble({
  message,
  index,
  labels,
  chatId,
}: {
  message: Message
  index: number
  labels: (typeof translations)[Language]
  chatId: string | null
}) {
  const user = message.role === 'user'
  const content = linkCitations(message.content, message.sources, message.id)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)
  const [feedbackBusy, setFeedbackBusy] = useState(false)
  const saveFeedback = async (rating: 'up' | 'down') => {
    if (typeof message.id !== 'number' || !chatId || feedbackBusy) return
    setFeedbackBusy(true)
    try {
      await api(`/chats/${chatId}/messages/${message.id}/feedback`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating }),
      })
      setFeedback(rating)
    } finally {
      setFeedbackBusy(false)
    }
  }

  return (
    <article className={`message-row ${user ? 'user-row' : 'assistant-row'}`}>
      {!user && <div className="assistant-sigil"><FileText size={14} /></div>}
      <div className={`message-content ${user ? 'user-content' : 'assistant-content'}`}>
        {!user && (
          <div className="answer-label">
            <span>{labels.brand}</span>
            {index === 0 && <span className="answer-label-line" />}
          </div>
        )}
        {user ? (
          <div className="user-bubble">{message.content}</div>
        ) : (
          <>
            <div className="answer-markdown">
              <ReactMarkdown
                components={{
                  pre: ({ children }) => <CodeBlock labels={labels}>{children}</CodeBlock>,
                  a: ({ href, children }) => {
                    return (
                    <a
                      href={href}
                      onClick={(event) => {
                        if (!href?.startsWith('#source-')) return
                        event.preventDefault()
                        const card = document.getElementById(href.slice(1))
                        const excerpt = card?.querySelector('details')
                        if (excerpt) excerpt.open = true
                        excerpt?.querySelector('summary')?.focus({ preventScroll: true })
                        card?.scrollIntoView({
                          behavior: 'smooth',
                          block: 'center',
                        })
                      }}
                    >
                      {children}
                    </a>
                    )
                  },
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
            {typeof message.id === 'number' && chatId && (
              <div className="answer-feedback" aria-label="Answer feedback">
                <button className={feedback === 'up' ? 'selected' : ''} onClick={() => void saveFeedback('up')} disabled={feedbackBusy}>
                  <ThumbsUp size={14} />{labels.helpful}
                </button>
                <button className={feedback === 'down' ? 'selected' : ''} onClick={() => void saveFeedback('down')} disabled={feedbackBusy}>
                  <ThumbsDown size={14} />{labels.notHelpful}
                </button>
                {feedback && <span role="status">{labels.feedbackSaved}</span>}
              </div>
            )}
            {message.content.trim() && !String(message.id).startsWith('stream-') && message.sources.length > 0 && (
              <section className="sources-panel" aria-label={labels.sources}>
                <div className="sources-heading">
                  <span><FileText size={14} />{labels.sources}</span>
                  <span>{message.sources.length}</span>
                </div>
                <div className="source-list">
                  {message.sources.map((source, sourceIndex) => (
                    <div
                      className="source-card"
                      id={sourceId(message.id, sourceIndex)}
                      key={`${source.source}-${source.page || 'doc'}-${source.chunk}`}
                    >
                      <span className="source-number">{sourceIndex + 1}</span>
                      <div className="source-copy">
                        <div className="source-name" dir="auto" title={source.source}>{source.source}</div>
                        <div className="source-actions">
                          {source.page && source.source.toLowerCase().endsWith('.pdf') && (
                            <a href={pdfPageUrl(source)} target="_blank" rel="noreferrer">
                              {labels.page} {source.page} ↗
                            </a>
                          )}
                        </div>
                        <details className="source-excerpt">
                          <summary>{labels.excerpt}</summary>
                          <p>{source.content}</p>
                        </details>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </article>
  )
}

export default App
