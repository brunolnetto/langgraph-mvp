import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'

interface Message {
  role: string
  content: string
  timestamp: string
}

const convElement = document.getElementById('conversation')!
const promptInput = document.getElementById('prompt-input') as HTMLInputElement
const spinner = document.getElementById('spinner')!

const SCROLL_TOLERANCE = 100

function saveToLocal(messages: Message[]) {
  localStorage.setItem('chat_history', JSON.stringify(messages))
}

function loadFromLocal(): Message[] {
  try {
    const raw = localStorage.getItem('chat_history')
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function renderMessage(message: Message): void {
  const { timestamp, role, content } = message
  const id = `msg-${timestamp}`

  let msgDiv = document.getElementById(id)
  if (!msgDiv) {
    msgDiv = document.createElement('div')
    msgDiv.id = id
    msgDiv.classList.add('border-top', 'pt-2', role)
    convElement.appendChild(msgDiv)
  }

  const readableTime = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  msgDiv.title = `${role} at ${timestamp}`
  msgDiv.innerHTML = `
    <div>${marked.parse(content)}</div>
    <small class="text-muted d-block mt-1 text-end">${readableTime}</small>
  `
}

function addMessages(responseText: string) {
  const lines = responseText.split('\n')
  const parsed: Message[] = lines.filter(l => l.length > 1).map(line => JSON.parse(line))
  const messages: Record<string, Message> = {}

  // dedup por timestamp, sobrescrevendo se necessário
  for (const msg of parsed) {
    messages[msg.timestamp] = msg
  }

  const sorted = Object.values(messages).sort((a, b) => a.timestamp.localeCompare(b.timestamp))
  sorted.forEach(renderMessage)
  saveToLocal(sorted)

  if (window.innerHeight + window.scrollY >= document.body.offsetHeight - SCROLL_TOLERANCE) {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  }
}

async function onFetchResponse(response: Response): Promise<void> {
  let text = ''
  const decoder = new TextDecoder()

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Unexpected response: ${response.status}\n${errorText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) return

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    text += decoder.decode(value)

    // Adiciona as mensagens enquanto o texto vai sendo carregado
    addMessages(text)

    // Adiciona um pequeno delay para garantir que o DOM seja atualizado
    await new Promise(resolve => setTimeout(resolve, 10));  // 10ms de delay
  }

  spinner.classList.remove('active')
  promptInput.disabled = false
  promptInput.focus()
}

function onError(error: any) {
  console.error(error);
  document.getElementById('error')?.classList.remove('d-none');
  spinner.classList.remove('active');
  const errorMessage = document.createElement('div');
  errorMessage.textContent = "Something went wrong. Please try again.";
  document.getElementById('error')?.appendChild(errorMessage);
}


async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  spinner.classList.add('active')

  const body = new FormData(e.target as HTMLFormElement)
  promptInput.value = ''
  promptInput.disabled = true

  const response = await fetch('/chat/', { method: 'POST', body })
  await onFetchResponse(response)
}

document.querySelector('form')!.addEventListener('submit', (e) => onSubmit(e).catch(onError))

// 🔁 Restaura histórico local + sincroniza com servidor
const cached = loadFromLocal()
cached.forEach(renderMessage)
fetch('/chat/').then(onFetchResponse).catch(onError)
