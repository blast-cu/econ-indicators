def form_prompts(texts: str, pre_prompt: str = '', post_prompt: str = ''):
    
    prompts = [pre_prompt + t[:1024] + post_prompt for t in texts]

    for p in prompts:
        print(p)
        print('\n')
    
    return prompts
