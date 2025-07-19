let mcqs = [];
let current = 0;

const questionEl = document.getElementById('question');
const optionsEl = document.getElementById('options');
const selectedOptionEl = document.getElementById('selected_option');

fetch('mcqs.json')
  .then(response => response.json())
  .then(data => {
    mcqs = data;
    loadQuestion();
  })
  .catch(err => {
    questionEl.textContent = "❌ Failed to load questions.";
    console.error(err);
  });

function loadQuestion() {
  const mcq = mcqs[current];
  questionEl.textContent = mcq.question;
  optionsEl.innerHTML = '';
  selectedOptionEl.innerHTML = '';

  for (let key in mcq.options) {
    const btn = document.createElement('div');
    btn.classList.add('option');
    btn.textContent = `${key}: ${mcq.options[key]}`;
    btn.dataset.option = key;
    btn.addEventListener('click', handleOptionClick);
    optionsEl.appendChild(btn);
  }
}

function handleOptionClick(e) {
  const selected = e.currentTarget.dataset.option;
  const correct = mcqs[current].answer;

  selectedOptionEl.innerHTML = `<strong>You selected:</strong> ${selected}`;

  if (selected === correct) {
    e.currentTarget.classList.add('correct');
    // Disable all buttons
    [...optionsEl.children].forEach(opt => opt.removeEventListener('click', handleOptionClick));

    setTimeout(() => {
      current++;
      if (current < mcqs.length) {
        loadQuestion();
      } else {
        questionEl.textContent = "🎉 Quiz Completed!";
        optionsEl.innerHTML = '';
        selectedOptionEl.innerHTML = '';
        setTimeout(() => window.close(), 2000);
      }
    }, 500);
  } else {
    e.currentTarget.classList.add('wrong');
    // Keep all options active so user can try again
  }
}
