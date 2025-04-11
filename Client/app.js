document.addEventListener('DOMContentLoaded', function() {
  const skillsInput = document.getElementById('skills');
  const dropdown = document.getElementById('suggestions-dropdown');
  const getCoursesBtn = document.getElementById('get-courses-btn');
  const courseList = document.getElementById('course-list');

  let allSkills = []; // This will hold the skills fetched from the backend

  const baseUrl = "https://ey-2-517h.onrender.com/"; // Base URL for API

  // Fetch skills from the backend on page load
  fetchSkills();

  // Function to fetch skills from the Flask backend
  async function fetchSkills() {
    const url = `${baseUrl}/get_skills`;  // Endpoint URL for skills
    try {
      const response = await fetch(url);
      const data = await response.json();
      allSkills = data.skills; // Store the skills
    } catch (error) {
      console.error('Error fetching skills:', error);
    }
  }
  document.addEventListener("DOMContentLoaded", function () {
    const dropdownBtn = document.querySelector(".dropbtn1");
    const dropdownContent = document.querySelector(".dropdown-content1");
  
    dropdownBtn.addEventListener("click", function (event) {
      event.stopPropagation(); // Prevents click from closing immediately
      dropdownContent.style.display =
        dropdownContent.style.display === "block" ? "none" : "block";
    });
    function toggleDropdown() {
    document.getElementById("dropdownMenu").classList.toggle("show");
  }
  
    // Close dropdown when clicking outside
    document.addEventListener("click", function (event) {
      if (!dropdownBtn.contains(event.target) && !dropdownContent.contains(event.target)) {
        dropdownContent.style.display = "none";
      }
    });
  });
  // Function to filter skills based on user input
  skillsInput.addEventListener('input', function() {
    const query = skillsInput.value.toLowerCase();
    dropdown.innerHTML = ''; // Clear previous suggestions

    if (query.length > 0) {
      const filteredSkills = allSkills.filter(skill => skill.toLowerCase().startsWith(query));

      // Show matching skills in the dropdown
      filteredSkills.forEach(skill => {
        const div = document.createElement('div');
        div.textContent = skill;
        div.onclick = () => {
          skillsInput.value = skill; // Set input to the selected skill
          dropdown.innerHTML = ''; // Clear dropdown on selection
        };
        dropdown.appendChild(div);
      });

      dropdown.style.display = filteredSkills.length > 0 ? 'block' : 'none';
    } else {
      dropdown.style.display = 'none'; // Hide dropdown when input is empty
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    const user = JSON.parse(localStorage.getItem("user")); // Check if user is logged in
    const findSkills = document.getElementById("findSkills");
    const recommendedCourses = document.getElementById("recommendedCourses");
  
    function restrictAccess(button) {
      button.addEventListener("click", function (event) {
        if (!user) {
          event.preventDefault(); // Stop navigation
          alert("You need to sign in first!");
          window.location.href = "signin.html"; // Redirect to sign-in page
        }
      });
    }
  
    restrictAccess(findSkills);
    restrictAccess(recommendedCourses);
  });
  

  // Button click event to fetch recommended courses based on the selected skill
  getCoursesBtn.addEventListener('click', async function() {
    const inputSkills = skillsInput.value.trim();

    if (inputSkills) {
      const url = `${baseUrl}/predict`; // Endpoint URL for course prediction
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ input: inputSkills })  // Send the input skills
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Display the recommended courses
        if (data.length > 0) {
          const ul = document.createElement('ul');
          data.forEach(course => {
            const li = document.createElement('li');
          
          const a = document.createElement('a');
          a.href = course.URL;
          a.textContent = course.Title;
          a.target = '_blank'; // Opens link in a new tab
          li.appendChild(a);
          ul.appendChild(li);
          
          });
          courseList.innerHTML = ''; // Clear previous results
          courseList.appendChild(ul);
        } else {
          courseList.innerHTML = 'No courses found.';
        }
      } catch (error) {
        console.error('Error fetching recommended courses:', error);
        courseList.innerHTML = 'Error fetching courses. Please check the console for more details.';
      }
    } else {
      alert('Please enter a skill.');
    }
  });
});
