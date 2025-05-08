// Definindo interfaces e tipos
interface Message {
  role: string;
  content: string;
  timestamp: string;
}

// Definindo os elementos do DOM de forma mais segura
const conversationElement = document.getElementById('conversation') as HTMLElement | null;
const promptInput = document.getElementById('prompt-input') as HTMLInputElement | null;
const form = document.getElementById('chat-form') as HTMLFormElement | null;
const sendButton = document.getElementById('send-button') as HTMLButtonElement | null;
const sendIcon = document.getElementById('send-icon') as HTMLElement | null;
const spinnerIcon = document.getElementById('spinner-icon') as HTMLElement | null;
const thinkingElement = document.getElementById('thinking') as HTMLElement | null;

// Verificando se os elementos essenciais existem
if (!conversationElement || !promptInput || !form || !sendButton || !sendIcon || !spinnerIcon || !thinkingElement) {
  console.error('Critical DOM elements are missing. Please check your HTML.');
}

// Função para alternar o tema
function toggleTheme(): void {
  const body = document.body;
  const icon = document.querySelector('#theme-icon') as HTMLElement | null;

  if (!icon) {
    console.warn('Theme toggle icon not found.');
    return;
  }

  body.classList.toggle('dark-mode');

  if (body.classList.contains('dark-mode')) {
    icon.className = 'fas fa-moon';
    localStorage.setItem('theme', 'dark');
  } else {
    icon.className = 'fas fa-sun';
    localStorage.setItem('theme', 'light');
  }
}

// Função para carregar o tema salvo
function loadTheme(): void {
  const savedTheme = localStorage.getItem('theme');
  const themeToggle = document.getElementById('theme-toggle') as HTMLElement | null;
  const themeIcon = document.getElementById('theme-icon') as HTMLElement | null;

  if (!themeToggle || !themeIcon) {
    console.warn('Theme toggle elements not found.');
    return;
  }

  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
    themeIcon.className = 'fas fa-moon';
  }

  themeToggle.addEventListener('click', toggleTheme);
}

// Função para rolar a página para o fundo - melhorada com forceScroll opcional
function scrollToBottom(forceScroll: boolean = false): void {
  if (!conversationElement) {
    console.warn('Conversation element not found, unable to scroll.');
    return;
  }
  
  // Sempre aplica o scroll imediatamente
  conversationElement.scrollTop = conversationElement.scrollHeight;
  
  // Se forceScroll for true, também agenda um segundo scroll após o renderizador
  if (forceScroll) {
    setTimeout(() => {
      if (conversationElement) {
        conversationElement.scrollTop = conversationElement.scrollHeight;
      }
    }, 10);
  }
}

// Função para exibir o indicador de "pensando"
function showThinking(show: boolean): void {
  if (!thinkingElement) {
    console.warn('Thinking element not found.');
    return;
  }
  
  if (show) {
    thinkingElement.classList.remove('d-none');
    scrollToBottom(true);
  } else {
    thinkingElement.classList.add('d-none');
  }
}

// Criar elemento de mensagem do usuário
function createUserMessageElement(content: string): HTMLElement {
  const wrapper = document.createElement('div');
  wrapper.classList.add('message-wrapper');
  wrapper.innerHTML = `
    <div class="user">
      <div class="message-header">
        <i class="fas fa-user"></i>You
      </div>
      <div class="message-content">${content}</div>
    </div>
  `;
  return wrapper;
}

// Criar elemento de mensagem do assistente
function createAssistantMessageElement(): HTMLElement {
  const wrapper = document.createElement('div');
  wrapper.classList.add('message-wrapper');
  
  // Usar um ID único para cada mensagem do assistente
  const messageId = `assistant-message-${Date.now()}`;
  
  wrapper.innerHTML = `
    <div class="model">
      <div class="message-header">
        <i class="fas fa-robot"></i>AI Assistant
      </div>
      <div id="${messageId}" class="message-content"></div>
    </div>
  `;
  return { wrapper, messageId };
}

// Função para adicionar mensagem do usuário
function addUserMessage(userMessage: string): void {
  if (!conversationElement || !thinkingElement) {
    console.error('Unable to add message: conversation or thinking element not found.');
    return;
  }

  const userMessageElem = createUserMessageElement(userMessage);
  conversationElement.insertBefore(userMessageElem, thinkingElement);
  scrollToBottom(true);
}

// Função para alternar o estado da UI (habilitar/desabilitar campos)
function toggleUIState(isBusy: boolean): void {
  if (promptInput && sendButton && sendIcon && spinnerIcon) {
    promptInput.disabled = isBusy;
    sendButton.disabled = isBusy;
    
    if (isBusy) {
      sendIcon.classList.add('d-none');
      spinnerIcon.classList.remove('d-none');
    } else {
      sendIcon.classList.remove('d-none');
      spinnerIcon.classList.add('d-none');
    }
  } else {
    console.error('UI elements missing for state toggle.');
  }
}

// Função para manipular a submissão do formulário
async function setupFormHandler(): Promise<void> {
  if (!form || !promptInput) {
    console.error('Form or prompt input element not found.');
    return;
  }

  form.addEventListener('submit', async (e: SubmitEvent) => {
    e.preventDefault();

    const userMessage = promptInput.value.trim();
    if (!userMessage) return;

    // Desabilitar UI durante o processamento
    toggleUIState(true);
    
    // Adicionar mensagem do usuário
    addUserMessage(userMessage);
    
    // Mostrar indicador "pensando"
    showThinking(true);

    try {
      // Limpar o campo de entrada logo após enviar
      promptInput.value = '';
      
      // Preparar elemento para resposta do assistente
      const { wrapper: assistantElem, messageId } = createAssistantMessageElement();
      
      // Inicialmente oculta a mensagem do assistente até que o conteúdo comece a chegar
      if (conversationElement && thinkingElement) {
        conversationElement.insertBefore(assistantElem, thinkingElement);
      }
      
      const assistantMessageContent = document.getElementById(messageId);
      if (!assistantMessageContent) {
        console.error('Assistant message content element not found.');
        return;
      }

      // Fazer a requisição para o backend
      const response = await fetch('/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMessage }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Unexpected response: ${response.status}`);
      }

      // Processar stream de resposta
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let assistantResponseReceived = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const msg = JSON.parse(line);
            
            // Se for mensagem do modelo (assistente)
            if (msg.role === 'model') {
              assistantResponseReceived = true;
              // Remover o indicador de pensando assim que começarmos a receber conteúdo
              if (assistantMessageContent.textContent === '') {
                showThinking(false);
              }
              
              assistantMessageContent.textContent = msg.content;
              scrollToBottom();
            }
          } catch (err) {
            console.warn('Invalid JSON chunk:', line, err);
          }
        }
      }

      // Se por algum motivo não recebemos nenhuma resposta do assistente
      if (!assistantResponseReceived) {
        assistantMessageContent.textContent = "Sorry, I couldn't process your request.";
      }
      
      // Esconder indicador de pensando e rolar para o fundo
      showThinking(false);
      scrollToBottom(true);
    } catch (error) {
      console.error('Error while streaming:', error);
      const errorElement = document.getElementById('error');
      if (errorElement) errorElement.classList.remove('d-none');
      showThinking(false);
    } finally {
      toggleUIState(false);
      promptInput.focus();
    }
  });
}

// Carregar mensagens existentes
async function loadExistingMessages(): Promise<void> {
  try {
    const response = await fetch('/chat/');
    if (!response.ok) {
      throw new Error(`Failed to load messages: ${response.status}`);
    }
    
    const text = await response.text();
    const lines = text.split('\n').filter(line => line.trim());
    
    if (lines.length === 0) {
      return; // Nenhuma mensagem para carregar
    }
    
    // Ocultar o indicador de pensando enquanto carregamos mensagens
    showThinking(false);
    
    let lastUserMessage = '';
    
    for (const line of lines) {
      try {
        const msg = JSON.parse(line);
        
        if (msg.role === 'user') {
          lastUserMessage = msg.content;
          addUserMessage(msg.content);
        } else if (msg.role === 'model' && lastUserMessage) {
          const { wrapper: assistantElem, messageId } = createAssistantMessageElement();
          
          if (conversationElement && thinkingElement) {
            conversationElement.insertBefore(assistantElem, thinkingElement);
            
            // Atualizar o conteúdo após adicionar ao DOM
            const messageContent = document.getElementById(messageId);
            if (messageContent) {
              messageContent.textContent = msg.content;
            }
          }
        }
      } catch (err) {
        console.warn('Failed to parse message:', line, err);
      }
    }
    
    // Rolagem final após carregar todas as mensagens
    scrollToBottom(true);
  } catch (error) {
    console.error('Error loading messages:', error);
  }
}

// Inicializar o app
window.addEventListener('DOMContentLoaded', () => {
  loadTheme();
  setupFormHandler();
  
  // Inicialmente ocultar o indicador de pensando 
  showThinking(false);
  
  // Carregar mensagens existentes
  loadExistingMessages().then(() => {
    // Se não houver mensagens após o carregamento, adicionar uma mensagem de boas-vindas
    setTimeout(() => {
      if (conversationElement && conversationElement.children.length <= 1) {
        addUserMessage("Hello! How can you help me?");
        
        // Criar uma resposta de boas-vindas
        const { wrapper: welcomeElem, messageId } = createAssistantMessageElement();
        if (conversationElement && thinkingElement) {
          conversationElement.insertBefore(welcomeElem, thinkingElement);
          
          const welcomeContent = document.getElementById(messageId);
          if (welcomeContent) {
            welcomeContent.textContent = "Hi there! I'm your AI assistant. Feel free to ask me anything!";
          }
          
          scrollToBottom(true);
        }
      }
    }, 500);
  });
});

// Garantir que o scroll siga o conteúdo
if (conversationElement) {
  const observer = new MutationObserver(() => scrollToBottom());
  observer.observe(conversationElement, { childList: true, subtree: true });
}