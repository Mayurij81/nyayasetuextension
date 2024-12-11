css = '''
<style>
.chat-message {
  padding: 0.5rem 1rem; /* Adjust padding for consistent spacing */
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  display: flex;
   /* Limit the width of the message container */

}
.chat-message.user {

  background-color: #bcebe6;
  margin-left: auto; /* Shift the entire message block to the right */
  text-align: right; /* Align text to the right */
  display: flex; /* Ensures image and message stay aligned horizontally */
  justify-content: flex-end; /* Align content to the right within the flex container */
  align-items: center; /* Vertically align the avatar and message */
  flex-direction: row-reverse;
  width: auto;
  max-width: 40%; /* Let the container adjust to the content */

}
.chat-message.bot {
  background-color: #e9eaec;
  margin-right: auto; /* Align the bot container to the left */
  width: auto; /* Let the container adjust to the content */
  max-width: 75%;
  
}
.chat-message .avatar {
  margin-left: 10px; /* Space between text and avatar */
  width: 20%;
}
.chat-message .avatar img {
  max-width: 30px;
  max-height: 30px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 0.5remrem;
  color: #000000;
    word-wrap: break-word; /* Handle long words gracefully */

}

/* FAQ Button Styles */
.faq-container {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 1rem;
}

.faq-button {
  background-color: #475063;
  color: white;
  padding: 10px 15px;
  border: none;
  border-radius: 5px;
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.faq-button:hover {
  background-color: #2f3640;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <a href="https://imgbb.com/">
            <img src="https://i.ibb.co/1XLGTs1/Whats-App-Image-2024-11-21-at-20-26-32-5dec28a6.jpg" 
            alt="Bot Avatar" 
            style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
        </a>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <a href="https://imgbb.com/">
          <img src="https://i.ibb.co/RcCJZ48/Whats-App-Image-2024-11-23-at-00-28-22-8496dfbe.jpg" 
          alt="User Avatar" 
          style="max-height: 30px; max-width: 30px; border-radius: 50%; object-fit: cover;">
        </a>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

faq_button_template = '''
<div class="faq-container">
    {% for question in QUESTIONS %}
        <button class="faq-button" onclick="sendFAQQuestion('{{question}}')">{{question}}</button>
    {% endfor %}
</div>

<script>
function sendFAQQuestion(question) {
    const userInput = document.querySelector('input[aria-label="Ask a question"]');
    userInput.value = question;
    userInput.dispatchEvent(new Event('change', { bubbles: true }));
}
</script>
'''