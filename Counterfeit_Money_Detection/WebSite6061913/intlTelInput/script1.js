document.getElementById("login-form").addEventListener("submit", function(event) {
	event.preventDefault();
	let username = document.getElementById("username").value;
	let password = document.getElementById("password").value;
	if (username === "admin" && password === "password") {
		document.getElementById("welcome-message").textContent = "Welcome, " + username + "!";
	} else {
		document.getElementById("welcome-message").textContent = "Invalid username or password.";
	}
});