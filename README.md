This is News Search and Summarization Web Application where you will provide URLs of the articles in which you want to search something specific. Example: I provided URLs like CEO of Canadian bank etc and then I asked specific questions like who is the CEO of RBC?

Download the requirements.txt file: pip install -r requirements.txt
Create account on OpenAI and add your own API Key in .env file
You have to install python-magic-bin as well. Try using the pip command (pip install python-magic-bin). Since, I was facing issues with installing it with pip, so I used conda: conda install -c conda-forge libmagic. No need to install if you are on Linux
I used the OpenAI's gpt-3.5-turbo-instruct model but first I tried using the text-davinci-003 but this is deprecated and no longer available to use.
