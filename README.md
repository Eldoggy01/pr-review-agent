High-level view of the architecture of the system:

<img width="2200" height="1762" alt="image" src="https://github.com/user-attachments/assets/03f6a3ff-6acc-486e-ba95-b2a44b8c4eb9" />

We have three agents: ContextAgent to fetch details from a GitHub repository by PR number. CommentorAgent to create a draft comment. ReviewAndPostingAgent reviews the generated draft comment, check if refinements are needed, and finally post it to GitHub
