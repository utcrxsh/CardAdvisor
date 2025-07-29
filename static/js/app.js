// CardAdvisor Frontend JavaScript

class CardAdvisor {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.debugMode = document.getElementById('debugMode');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        this.init();
    }

    init() {
        // Focus on input when page loads
        this.messageInput.focus();
        
        // Add event listeners
        this.debugMode.addEventListener('change', this.toggleDebugMode.bind(this));
        
        // Auto-resize chat messages container
        this.resizeChatContainer();
        window.addEventListener('resize', this.resizeChatContainer.bind(this));
    }

    resizeChatContainer() {
        const chatContainer = document.querySelector('.chat-container');
        const navbarHeight = document.querySelector('.navbar').offsetHeight;
        chatContainer.style.height = `calc(100vh - ${navbarHeight}px)`;
    }

    async sendMessage(message = null) {
        const userMessage = message || this.messageInput.value.trim();
        
        if (!userMessage) return;

        // Clear input
        this.messageInput.value = '';

        // Add user message to chat
        this.addMessage(userMessage, 'user');

        // Show loading
        this.showLoading();

        try {
            // Send to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage })
            });

            const data = await response.json();

            // Hide loading
            this.hideLoading();

            if (data.error) {
                this.addMessage(`❌ ${data.error}`, 'assistant');
            } else {
                // Process and display response
                this.processResponse(data.response, data.details);
            }

        } catch (error) {
            this.hideLoading();
            this.addMessage(`❌ Error: ${error.message}`, 'assistant');
        }
    }

    processResponse(response, details = null) {
        // Check if response contains card recommendations
        if (response.includes('Top ') && response.includes('Recommendation')) {
            this.displayCardRecommendations(response, details);
        } else {
            // Regular text response
            this.addMessage(response, 'assistant', details);
        }
    }

    displayCardRecommendations(response, details) {
        const lines = response.split('\n');
        let cardSection = false;
        let currentCard = {};
        let cards = [];
        let recommendation = '';
        let summary = '';

        for (let line of lines) {
            if (line.startsWith('Summary of Top ')) {
                summary = line;
                cardSection = true;
            } else if (cardSection && /^\d+\.\s+/.test(line)) {
                if (Object.keys(currentCard).length > 0) {
                    cards.push(currentCard);
                }
                currentCard = { name: line.replace(/^\d+\.\s*/, '').trim() };
            } else if (cardSection && line.includes('- ')) {
                const [key, value] = line.split(':').map(s => s.trim());
                const cleanKey = key.replace('-', '').trim();
                currentCard[cleanKey] = value;
            } else if (line.startsWith('Recommendation:')) {
                if (Object.keys(currentCard).length > 0) {
                    cards.push(currentCard);
                }
                cardSection = false;
                recommendation = line + '\n' + lines.slice(lines.indexOf(line) + 1).join('\n');
                break;
            }
        }

        // Add the last card if exists
        if (Object.keys(currentCard).length > 0) {
            cards.push(currentCard);
        }

        // Create HTML for cards
        let cardsHtml = '';
        if (summary) {
            cardsHtml += `<h5 class="text-warning mb-3">${summary}</h5>`;
        }

        cards.forEach(card => {
            cardsHtml += `
                <div class="card-recommendation">
                    <h6 class="card-title">${card.name || 'Unknown Card'}</h6>
                    <div class="card-content">
                        ${card.Score ? `<span class="card-badge">Score: ${card.Score}</span>` : ''}
                        ${card['Annual Fee'] ? `<div><strong>Annual Fee:</strong> ${card['Annual Fee']}</div>` : ''}
                        ${card['Reward Rate'] ? `<div><strong>Reward Rate:</strong> ${card['Reward Rate']}</div>` : ''}
                        ${card['Lounge Access'] ? `<div><strong>Lounge Access:</strong> ${card['Lounge Access']}</div>` : ''}
                        ${card['Luxury Perks'] ? `<div><strong>Luxury Perks:</strong> ${card['Luxury Perks']}</div>` : ''}
                        ${card['Welcome Bonus'] ? `<div><strong>Welcome Bonus:</strong> ${card['Welcome Bonus']}</div>` : ''}
                    </div>
                </div>
            `;
        });

        if (recommendation) {
            cardsHtml += `
                <div class="card-recommendation">
                    <h6 class="card-title">Recommendation</h6>
                    <div class="card-content">
                        ${recommendation.replace(/\n/g, '<br>')}
                    </div>
                </div>
            `;
        }

        this.addMessage(cardsHtml, 'assistant', details, true);
    }

    addMessage(content, sender, details = null, isHtml = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (sender === 'assistant') {
            const iconDiv = document.createElement('div');
            iconDiv.className = 'd-flex align-items-center mb-2';
            iconDiv.innerHTML = '<i class="bi bi-credit-card-2-front text-warning me-2"></i><strong>CardAdvisor</strong>';
            messageContent.appendChild(iconDiv);
        }

        if (isHtml) {
            messageContent.innerHTML += content;
        } else {
            const textDiv = document.createElement('p');
            textDiv.className = 'mb-0';
            textDiv.textContent = content;
            messageContent.appendChild(textDiv);
        }

        // Add debug info if enabled
        if (this.debugMode.checked && details) {
            const debugDiv = document.createElement('div');
            debugDiv.className = 'debug-info mt-2';
            debugDiv.innerHTML = `<strong>Debug Info:</strong><br><pre>${JSON.stringify(details, null, 2)}</pre>`;
            messageContent.appendChild(debugDiv);
        }

        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showLoading() {
        this.loadingModal.show();
    }

    hideLoading() {
        this.loadingModal.hide();
    }

    toggleDebugMode() {
        // This will affect future messages
        console.log('Debug mode:', this.debugMode.checked);
    }

    askQuestion(question) {
        this.messageInput.value = question;
        this.sendMessage(question);
    }
}

function adjustChatPadding() {
    const chatMessages = document.querySelector('.chat-messages');
    const chatInput = document.querySelector('.chat-input-container');
    if (chatMessages && chatInput) {
        chatMessages.style.paddingBottom = (chatInput.offsetHeight + 24) + 'px'; // 24px extra for spacing
    }
}

window.addEventListener('resize', adjustChatPadding);
document.addEventListener('DOMContentLoaded', adjustChatPadding);

// Patch CardAdvisor to call adjustChatPadding after sending a message
const originalAddMessage = CardAdvisor.prototype.addMessage;
CardAdvisor.prototype.addMessage = function(...args) {
    originalAddMessage.apply(this, args);
    adjustChatPadding();
};

// Global functions for HTML onclick handlers
function sendMessage() {
    if (window.cardAdvisor) {
        window.cardAdvisor.sendMessage();
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function askQuestion(question) {
    if (window.cardAdvisor) {
        window.cardAdvisor.askQuestion(question);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.cardAdvisor = new CardAdvisor();
}); 