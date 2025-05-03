document.addEventListener('DOMContentLoaded', function() {
    // Get the form element
    const form = document.querySelector('form');
    
    // Add event listener for form submission
    if (form) {
        form.addEventListener('submit', function(event) {
            // Prevent the default form submission
            event.preventDefault();
            
            // Check if all required fields are filled
            const requiredFields = form.querySelectorAll('[required]');
            let allFieldsFilled = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    allFieldsFilled = false;
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                }
            });
            
            // If all fields are filled, submit the form
            if (allFieldsFilled) {
                console.log('Form is valid, submitting...');
                form.submit();
            } else {
                console.log('Form is invalid, please fill all required fields');
                // Focus on the first empty field
                const firstEmptyField = Array.from(requiredFields).find(field => !field.value.trim());
                if (firstEmptyField) {
                    firstEmptyField.focus();
                }
            }
        });
    }
    
    // Add event listeners for input fields to remove validation styling on input
    const inputFields = document.querySelectorAll('input, select');
    inputFields.forEach(field => {
        field.addEventListener('input', function() {
            if (field.value.trim()) {
                field.classList.remove('is-invalid');
            }
        });
    });
});
