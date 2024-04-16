const form = document.getElementById("signup-form");

form.addEventListener("submit", e => {
  e.preventDefault();

  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  // Here you can send the data to your server to create a new user

  console.log(`Name: ${name}, Email: ${email}, Password: ${password}`);

  // Reset the form
  form.reset();
});