preprocess_dict = {
    "regex_patterns": {
        r"(Updated at )?\d{1,2}[\.:]\d{2}\s(?:EST|UTC)(?:\s\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})?(?:\s\d{1,2}\.\d{2}\sEST)?",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}",
        r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\s\d{2}\.\d{2}\sEST",
        r"(From )?\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\s\d{4}\s\d{2}\.\d{2}\sEST)?",
        r"\d{1,2}[hm|(min)] ago",

        # Email instructions
        r"Write to [^@]+@\w+\.\w+(?:\sand\s[^@]+@\w+\.\w+)?",
        
        # Contributions and reports
        r"(?:Contributing:|With)?\s*(?:The\s)?(?:Associated\sPress|Agence\sFrance\-Presse|Reuters)(?:\s(?:and|&)?\s(?:The\s)?(?:Associated\sPress|Agence\sFrance\-Presse|Reuters))?(?:\scontributed\s(?:to\sthis\sreport|reporting))?\.?",

        
        # Time indicators
        r"\b(?:Around )?\d{1,2}:\d{2}\s(?:a\.m\.|p\.m\.)",
        
        # Listening times
        r"Listen\s\d+\smin",
        
        # Photograph credits
        r"Photograph:.*Updated\sat.*",

        r"This story has been updated( with additional information|developments|details)?.",

        # Social media tags
        r"^Follow [A-Za-z]+ [A-Za-z]+ on Twitter @\w+$"
    },

    "all": {  
        "End of carousel",
        "Find it all here",
        "skip past newsletter promotion",
        "This story has been updated",
        "Privacy Notice:",
        "Advertisement", 
        "Ad Feedback",
        "The Associated Press contributed to this report",
        "This is a developing story",
        "(ap)",
        "associated press",
        "reuters",
        "VIDEO:",
        "breaking news",
        "getty image",
        "your inbox",
        "The news and stories that matter, delivered weekday mornings",
        "mailto:",
        "Your browser does not support",
        "article is in your queue",
        "Image",
        "Credit",
        "Here's what to know",
        "Return to menu",
        "Photo:",
        "Copyright",
        "All Rights Reserved",
        "Last Updated:",
        "Newsletter Sign-up",
        "This article is in your queue",
        "Your browser does not support",
        "Send any friend a story",
        "As a subscriber",
        "This copy is for your personal, non-commercial use only",
        "Distribution and use of this material are governed by our Subscriber Agreement",
        "For non-personal use or to order multiple copies"
    },
    "breitbart": {
        "SUBSCRIBE for free by clicking your preferred podcast platform below."
    },
    "cnn": {
    "CNN —\n", 
    "( CNN )"
    },
    "foxnews": {
        "CLICK HERE TO GET THE FOX NEWS APP",
        "GET FOX BUSINESS ON THE GO BY CLICKING HERE",
        "CLICK HERE FOR MORE FOX NEWS OPINION",
        "WATCH MORE FOX NEWS DIGITAL ORIGINALS HERE",
        "CLICK HERE FOR MORE SPORTS COVERAGE ON FOXNEWS.COM",
        "CLICK HERE TO READ MORE ON FOX BUSINESS",
        "CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER",
        "LIKE WHAT YOU’RE READING? CLICK HERE FOR MORE ENTERTAINMENT NEWS",
        "Join Fox News for access to this content",
        "Read this article for free!",
        "Plus get unlimited access to thousands of articles, videos and more with your free account!",
        "Please enter a valid email address.",
        "By entering your email, you are agreeing to Fox News Terms of Service and Privacy Policy , which includes our Notice of Financial Incentive",
        "To access the content, check your email and follow the instructions provided.",
        "Plus special access to select articles and other premium content with your account - free of charge."
        },
    "huffpost": {
        "Advertisement",
        "Support HuffPost The Stakes Have Never Been Higher",
        "At HuffPost, we believe that everyone needs high-quality journalism, but we understand that not everyone can afford to pay for expensive news subscriptions. That is why we are committed to providing deeply reported, carefully fact-checked news that is freely accessible to everyone.",
        "Our News, Politics and Culture teams invest time and care working on hard-hitting investigations and researched analyses, along with quick but robust daily takes. Our Life, Health and Shopping desks provide you with well-researched, expert-vetted information you need to live your best life, while HuffPost Personal, Voices and Opinion center real stories from real people.",
        "This is why HuffPost's journalism is free for everyone, not just those who can afford expensive paywalls.",
        "We cannot do this without your help. Support our newsroom by contributing as little as $1 a month.",
        "Help keep news free for everyone by giving us as little as $1. Your contribution will go a long way.",
        "LOADING ERROR LOADING",
        "Support HuffPost",
        "Our 2024 Coverage Needs You",
        "Your Loyalty Means The World To Us",
        "Dear HuffPost Reader",
        "Contribute as little as $2 to keep our news free for all.",
        "Would you join us to help keep our stories free for all? Your contribution of as little as $2 will go a long way.",
        "Whether you come to HuffPost for updates on the 2024 presidential race, hard-hitting investigations into critical issues facing our country today, or trending stories that make you laugh, we appreciate you. The truth is, news costs money to produce, and we are proud that we have never put our stories behind an expensive paywall.",
        "Thank you for your past contribution to HuffPost. We are sincerely grateful for readers like you who help us ensure that we can keep our journalism free for everyone.",
        "Already contributed? Log in to hide these messages.",
        "The stakes are high this year, and our 2024 coverage could use continued support. Would you consider becoming a regular HuffPost contributor?"
        },
    "nytimes": {
        "Listen and follow The Daily",
        "Apple Podcasts | Spotify | Amazon Music",
        "The New York Times\n\n",
        "Advertisement",
        "The New York Times Audio app is home to journalism and storytelling, and provides news, depth and serendipity. If you haven’t already, download it here — available to Times news subscribers on iOS — and sign up for our weekly newsletter.",
        "The Headlines brings you the biggest stories of the day from the Times journalists who are covering them, all in about five minutes."
    },
    "theguardian": {
        "Get the day’s headlines and highlights emailed direct to you every morning",
        r"Sign up to (What's On|This is Europe|Headlines Europe|Down to Earth) Free (weekly )?newsletter",
        "Get the best TV reviews, news and exclusive features",
        r"Sign up to Headlines U[KS] Free newsletter",
        "Enter your email address Sign up Privacy Notice",
        "may contain info about charities, online ads, and content funded by outside parties",
        "Our morning email breaks down the key stories of the day, telling you what’s happening and why it matters",

    },
    "usatoday": {
        "What's everyone talking about? Sign up for our trending newsletter to get the latest news of the day",
        "SOURCE USA TODAY Network reporting and research",
    },
    "washingtonpost": {
        "Share Comment on this story Comment", 
        "Add to your saved stories Save", 
        "Share this article Share", 
        "Story continues below advertisement", 
        "Advertisement",
        "SCROLL TO CONTINUE",
        "About this story"
    },
    "wsj": {
        "Advertisement - Scroll to Continue",
        "Scroll to Continue"
        "Advertisement",
        "SHARE YOUR THOUGHTS",
        "Join the conversation below.",
        "Sign up for the WSJ Travel newsletter for more tips and insights from the Journal’s travel team.",
        "Sign up for the WSJ Workout Challenge to boost your fitness.",
        "—For more WSJ Technology analysis, reviews, advice and headlines, sign up for our weekly newsletter."
    }
}