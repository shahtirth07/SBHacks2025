const inputField = document.getElementById("user-input");
const chatWindow = document.getElementById("chat-window");

const sendMessage = async () => {
  const userInput = inputField.value.trim(); // Get input and trim whitespace
  if (!userInput) {
    console.log("No input to send"); // Debug log for empty input
    return;
  }

  // Clear the input field immediately
  inputField.value = "";
  inputField.focus(); // Ensure the cursor stays in the input field

  // Append the user's message to the chat window
  const userMessage = `<div><strong>You:</strong> ${userInput}</div>`;
  chatWindow.innerHTML += userMessage;

  try {
    // Send the user's message to the backend
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userInput }),
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    // Parse and display the bot's response
    const data = await response.json();
    const botResponse = `<div><strong>Bot:</strong><br>${data.response
      .split("\n")
      .map((line) => `${line}`)
      .join("<br>")}</div>`;
    chatWindow.innerHTML += botResponse;
  } catch (error) {
    // Display any error messages in the chat window
    chatWindow.innerHTML += `<div><strong>Error:</strong> ${error.message}</div>`;
  } finally {
    // Scroll the chat window to the bottom after rendering
    setTimeout(() => {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }, 0);
  }
};

// Add event listeners for Enter key and Send button
inputField.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    console.log("Enter key pressed"); // Debug log
    sendMessage();
  }
});
inputField.value = ""; // Clear input field

document.getElementById("send-btn").addEventListener("click", () => {
  console.log("Send button clicked"); // Debug log
  sendMessage();
});

document.addEventListener("DOMContentLoaded", () => {
  const chatToggleBtn = document.getElementById("chat-toggle-btn");
  const chatContainer = document.getElementById("chat-container");

  chatToggleBtn.addEventListener("click", () => {
    // Toggle the chat container's visibility
    if (
      chatContainer.style.display === "none" ||
      chatContainer.style.display === ""
    ) {
      chatContainer.style.display = "flex"; // Show the chat container
    } else {
      chatContainer.style.display = "none"; // Hide the chat container
    }
  });
});
