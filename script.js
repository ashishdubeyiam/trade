document.addEventListener('DOMContentLoaded', () => {
    const createAssignmentForm = document.getElementById('create-assignment-form');
    const joinAssignmentForm = document.getElementById('join-assignment-form');
    const submitCodeButton = document.getElementById('submit-code');
    const assignmentDetailsDiv = document.getElementById('assignment-details');
    const assignmentTitleH3 = document.getElementById('assignment-title');
    const assignmentDescriptionP = document.getElementById('assignment-description');
    const assignmentLanguageSpan = document.getElementById('assignment-language');
    const studentCodeTextarea = document.getElementById('student-code');
    const outputPre = document.getElementById('output');
    const errorAreaDiv = document.getElementById('error-area');
    const errorsPre = document.getElementById('errors');

    let currentAssignment = null;

    // Teacher Portal: Create Assignment
    if (createAssignmentForm) {
        createAssignmentForm.addEventListener('submit', (event) => {
            event.preventDefault();
            const title = event.target['project-title'].value;
            const description = event.target['project-description'].value;
            const language = event.target['programming-language'].value;
            const assignmentCode = event.target['assignment-code'].value;

            // In a real application, this would be sent to a backend
            console.log('New Assignment Created (simulated):');
            console.log({ title, description, language, assignmentCode });
            alert(`Assignment "${title}" created with code: ${assignmentCode}. Tell students to use this code.`);

            // Store globally for student simulation (very basic)
            // In a real app, students would fetch this from a server using the code
            window.assignments = window.assignments || {};
            window.assignments[assignmentCode] = { title, description, language };

            createAssignmentForm.reset();
        });
    }

    // Student Portal: Join Assignment
    if (joinAssignmentForm) {
        joinAssignmentForm.addEventListener('submit', (event) => {
            event.preventDefault();
            const studentName = event.target['student-name'].value;
            const joinCode = event.target['join-code'].value;

            console.log(`${studentName} attempting to join with code: ${joinCode}`);

            // Simulate fetching assignment details
            // In a real app, this would be an API call
            if (window.assignments && window.assignments[joinCode]) {
                currentAssignment = window.assignments[joinCode];
                assignmentTitleH3.textContent = currentAssignment.title;
                assignmentDescriptionP.textContent = currentAssignment.description;
                assignmentLanguageSpan.textContent = currentAssignment.language;

                assignmentDetailsDiv.style.display = 'block';
                outputPre.textContent = '';
                errorsPre.textContent = '';
                errorAreaDiv.style.display = 'none';
                studentCodeTextarea.value = ''; // Clear previous code

                alert(`Joined assignment: ${currentAssignment.title}`);
            } else {
                alert('Invalid assignment code. Please check and try again.');
                assignmentDetailsDiv.style.display = 'none';
                currentAssignment = null;
            }
            joinAssignmentForm.reset();
        });
    }

    // Student Portal: Submit Code
    if (submitCodeButton) {
        submitCodeButton.addEventListener('click', () => {
            if (!currentAssignment) {
                alert('Please join an assignment first.');
                return;
            }

            const code = studentCodeTextarea.value;
            console.log('Submitting code for assignment:', currentAssignment.title);
            console.log('Language:', currentAssignment.language);
            console.log('Code:', code);

            // Simulate online compiler interaction
            outputPre.textContent = ''; // Clear previous output
            errorsPre.textContent = ''; // Clear previous errors
            errorAreaDiv.style.display = 'none';

            // --- Placeholder for Online Compiler API Integration ---
            // This is where you would make an API call to an online compiler service
            // For now, we'll simulate a successful execution or an error

            if (code.trim() === '') {
                errorsPre.textContent = 'Error: Code cannot be empty.';
                errorAreaDiv.style.display = 'block';
                outputPre.textContent = '';
                return;
            }

            // Simulate different outputs based on language (very basic)
            if (currentAssignment.language === 'python') {
                if (code.includes('print')) {
                    outputPre.textContent = `Simulated Python Output for: ${code.substring(0, 50)}...`;
                } else {
                    errorsPre.textContent = 'Simulated Python Error: Missing print statement (example).';
                    errorAreaDiv.style.display = 'block';
                }
            } else if (currentAssignment.language === 'java') {
                 if (code.includes('System.out.println')) {
                    outputPre.textContent = `Simulated Java Output for: ${code.substring(0,50)}...`;
                } else {
                    errorsPre.textContent = 'Simulated Java Error: Missing System.out.println (example).';
                    errorAreaDiv.style.display = 'block';
                }
            } else if (currentAssignment.language === 'html') {
                // For HTML, "output" could be rendering the HTML, or showing the code itself
                outputPre.textContent = `Simulated HTML "run":
${code}`;
            } else {
                outputPre.textContent = `Simulated output for ${currentAssignment.language}:
${code}`;
            }
            // --- End Placeholder ---

            if (outputPre.textContent) {
                 alert('Code submitted and "run" (simulated). Check the output area.');
            } else if (errorsPre.textContent) {
                 alert('Code submitted and "run" (simulated) with errors. Check the error area.');
            }
        });
    }
});
