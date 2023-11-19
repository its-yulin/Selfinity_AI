# Selfinity_AI: Personal Database
<img width="800" alt="Screenshot 2023-11-19 at 6 39 56 AM" src="https://github.com/its-yulin/Selfinity_AI/assets/79822409/dffe7486-2196-42f4-ad0b-0b23b5a73be1">
<img width="800" alt="Screenshot 2023-11-19 at 7 24 56 AM" src="https://github.com/its-yulin/Selfinity_AI/assets/79822409/38b53e57-8853-4f96-ad8d-3268992c7037">

## 💡 Inspiration
Our project was inspired by a critical realization that resonated in both our personal and professional realms – the challenge of efficiently managing and accessing data. 

In our personal lives, valuable hours that could be dedicated to loved ones are often lost in managing emails, social media, and numerous digital tasks. This universal challenge led us to envision a unified, AI-driven tool designed to streamline the management of digital information. Our goal was to create an assistant that simplifies these tasks and enhances personal life by efficiently organizing everyday digital interactions. Our vision was to develop a solution that could help reclaim these lost hours, transforming them into productive and meaningful time by managing personal data more efficiently.

In the professional world, we noticed a substantial drain on resources: employees frequently spend a significant amount of time navigating between various files and knowledge databases. This inefficiency is not only time-consuming but also costly. The onboarding process for new employees often translates into thousands to tens of thousands of dollars in training expenses and diverted attention from other staff. This investment continues until the employee becomes fully comfortable and competent in their role. The indirect costs, including the time senior employees spend explaining processes instead of focusing on their primary tasks, further amplify these expenses.

## 💎 What it does
Our AI assistant serves as a comprehensive digital life co-pilot, redefining the way we interact with our digital lives. Its capabilities go beyond mere data consolidation; it actively assists in managing and interpreting information:

- **Diverse Service Integration**: It connects with a range of services, including financial platforms through Plaid and email services via the Gmail API. This integration allows the assistant to pull relevant data from these services directly, providing a holistic view of your financial and communication landscapes.
- **Manual File Uploads**: For data not automatically captured through service integrations, the assistant offers a manual upload feature. This allows you to directly upload files such as PDFs, documents, and other important data, ensuring that all your information can be centralized and accessed through the assistant.
- **Automated Browsing Data Collection**: With your permission, the assistant can automatically scrape your browsing history and bookmarks. This feature is designed to gather data from your online activities, providing insights and summaries of the websites and resources you frequent. It’s an optional feature that you can enable or disable based on your privacy preferences.
User-Driven Customization: The level of integration and data collection is entirely under your control. You can choose which services to connect, what files to upload, and whether to enable automatic data scraping. This flexibility ensures that your assistant is tailored to your specific needs and comfort with data sharing.
- **Document Summarization and Connection Building**: The assistant provides concise summaries and allows you to interact with you via chatbot interface. It intelligently links concepts and ideas across various articles and mediums of information, drawing connections that might otherwise go unnoticed, thereby enriching the user's understanding and knowledge base. If you prefer to view them in a more standardized way, a customizable dashboard view will conveniently display your information.
- **Email Management**: It sifts through your inbox, identifying and summarizing emails that require immediate attention. This feature not only saves time but ensures that important communications are not overlooked in the daily deluge of emails.
Calendar Management: The assistant keeps track of your schedule, bringing up calendar invites and reminders so that no important event or meeting is missed. This proactive approach to time management helps users stay organized and punctual.
- **Financial Insight**: By analyzing your banking data, the AI assistant provides insights into your spending habits. It can identify trends, suggest budgets, and offer advice on financial decisions. This feature aims to promote better financial management and awareness, allowing users to make more informed choices about their spending.

In essence, our AI assistant is not just a tool for data organization; it's an intelligent companion designed to optimize your digital interaction, saving time and reducing stress. By automating mundane tasks and providing insightful analysis, it allows users to focus on what truly matters, enhancing both professional efficiency and personal life quality.

## 🛠 How we built it
Building our AI assistant was a complex process that combined state-of-the-art technologies and innovative methodologies:

- **Data Compilation Pipeline**: We began with a Python pipeline to compile and collect diverse data sources, including browsing history, emails, and financial transactions. This step was crucial for creating a unified data format for further processing.
- **Pinecone Vector Database Integration**: Post compilation, the data was processed and embedded into a Pinecone vector database, chosen for its efficiency in managing large volumes of vectorized data.
- **Embedding with OpenAI's Ada Model**: For embedding the data, we utilized OpenAI's Ada model. Ada's balance of performance and efficiency made it ideal for embedding our varied datasets while maintaining contextual relevance and accuracy.
- **Retrieval Augmented Generation Using GPT-4 Turbo**: At the core of our AI assistant's intelligence was the Retrieval Augmented Generation, powered by OpenAI's GPT-4 Turbo model. This approach enabled dynamic interaction with the embedded data, ensuring accurate and contextually relevant responses.
- **Front End Development with Jinja2**: We designed the user interface using Jinja2, a flexible and designer-friendly templating language for Python, enabling us to create a responsive and interactive front end.
- **Back End Development with FastAPI**: FastAPI was our choice for the back end development. This modern, fast web framework was instrumental in building APIs with Python, ensuring efficient request handling and seamless integration with the front end.
- **LangChain for Language Model Integration**: We incorporated LangChain, a framework specifically designed to facilitate the creation of applications using large language models. LangChain was instrumental in enhancing our assistant's capabilities in document analysis, summarization, chatbot functionalities, and code analysis.
- **API Integrations**: Our system integrated APIs like Gmail and Plaid for accessing email and financial data. We also created custom scripts for scraping browsing history and bookmarks, enriching the assistant's data pool.
- **Security and Privacy Protocols**: Given the sensitive nature of the data, we adhered to strict security and privacy protocols. All data handling and storage complied with top-tier data protection standards.
- **User-Centric Interface and Experience**: The user interface, powered by Jinja2, was crafted with a focus on simplicity and intuitiveness. Features like a customizable dashboard and a chatbot interface made our assistant accessible and easy to use for a diverse user base.

The development of this AI assistant represented a harmonious integration of technological innovation and user-centric design, resulting in a tool that is advanced, secure, efficient, and user-friendly.

## ️😰 Challenges we ran into
Our journey in building the AI assistant was marked by a diverse array of challenges, each contributing to the project's depth and sophistication:
- **Integrating Diverse Data Types**: A major challenge was integrating various data types into a cohesive system. This task was particularly complex as we navigated the technical demands while supporting team members experiencing their first hackathon.
Optimization for GPT-4 Turbo Model: Tailoring the system to work efficiently with the GPT-4 Turbo model requires detailed and iterative optimization. Achieving high efficiency and accuracy in responses was a nuanced and ongoing process.
- **Connecting Multiple Services**: We faced significant technical hurdles in integrating several services. This included the Gmail and Plaid APIs for accessing email and financial data, Pinecone for vector database management, LangChain for leveraging language models, and the OpenAI API for advanced AI functionalities. Each of these integrations was critical to ensure a comprehensive functionality and seamless user experience for our assistant.
- **Handling Large Arrays of Data and Token Usage**: Managing a vast volume of data and optimizing token usage for data processing and API interactions posed a substantial challenge. Ensuring efficient data handling was crucial for system stability and responsiveness.
Navigating through these challenges, from intricate technical integrations to fostering a collaborative and educational team environment, drove us to innovate and creatively solve problems. The culmination of these efforts was the creation of a robust, versatile AI assistant.

## ⭐️ Accomplishments that we're proud of
We take immense pride in our achievements with this project, especially considering the diversity of our team, which included many beginners and first-time hackathon participants. We successfully integrated a multitude of data types into a unified AI assistant that provides accurate and relevant responses. The ability of our system to handle a broad spectrum of queries with precision and efficiency stands as a testament to the robust integration of OpenAI's technologies with our innovative data processing techniques. This accomplishment is particularly noteworthy given the varying levels of experience within our team, highlighting our collective growth and learning throughout this project.

## 🔮 What's next
Our vision for the AI assistant extends beyond its current capabilities, as we aim to develop it into a commercially viable product that appeals to both B2B and B2C markets. This ambitious goal involves a series of strategic steps:

- **Iterative Development and User Feedback**: We plan to engage in numerous iterations and extensive user interviews. This process is vital for refining our understanding of the value proposition and ensuring that our product aligns with the needs and expectations of our users.
Expanding Functionalities: To enhance user experience and utility, we will introduce new features such as:
- **YouTube Video Scraping**: For curating and summarizing content from video platforms.
Integration with Fitness and Health Apps: Syncing data from fitness trackers and health apps to provide a comprehensive overview of the user's physical activity, dietary habits, sleep patterns, and overall wellness.
- **Auto-generated To-Do Lists**: To assist in organizing and prioritizing tasks based on user activity and preferences.
- **Automatic Responses and Task Completions**: Leveraging AI to respond to routine queries and perform simple tasks automatically, enhancing productivity.
- **Media Suggestions**: Offering recommendations on books, articles, videos, and other media based on user interests and interactions.
- **Messenger Compilations**: Aggregating messages from various platforms into a single, easily manageable interface.
- **Product-Market Fit**: A key focus will be on finding the product-market fit that effectively addresses the pain points of our customers, improving outcomes in their personal and professional lives.
- **User-Centric Approach**: By continuously iterating based on user feedback and adapting our strategies, we aim to make our assistant a solution that not only manages digital information but also provides meaningful and practical assistance.

Our journey ahead involves adapting and innovating to create a product that is technologically sophisticated, intuitive, and impactful. The goal is to make our AI assistant a fundamental part of users' lives, enriching their experiences through advanced information management and insightful assistance, tailored to their unique needs and preferences.

