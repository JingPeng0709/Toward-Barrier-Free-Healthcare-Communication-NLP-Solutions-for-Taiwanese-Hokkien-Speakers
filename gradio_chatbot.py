import gradio as gr
from model_utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.emerald, secondary_hue=gr.themes.colors.pink)) as demo:
    gr.Markdown('''<div style="text-align:center"><h1>Demo：</h1></div>''')
    gr.Markdown('''<div style="text-align:center"><h2>研究生：吳景鵬</h2></div>''')
    
    # gr.Markdown('''<div style="text-align:center"><h2>RAG</h2></div>''')
    # with gr.Row():
    #     with gr.Column(scale=0.33):
    #         gr.Markdown('''<div style="text-align:center"><h2>華文</h2></div>''')
    #         chatbot = gr.Chatbot([], elem_id="chatbot", height=250)
    #     with gr.Column(scale=0.33):
    #         gr.Markdown('''<div style="text-align:center"><h2>台漢</h2></div>''')
    #         translated_chatbot = gr.Chatbot([], elem_id="translated_chatbot", height=250)
    #     with gr.Column(scale=0.33):
    #         gr.Markdown('''<div style="text-align:center"><h2>台羅</h2></div>''')
    #         translated_chatbot_tl = gr.Chatbot([], elem_id="translated_chatbot_tl", height=250)
    
        # with gr.Column(scale=0.33):
        #     gr.Markdown('''<div style="text-align:center"><h2>Chatbot(台羅)</h2></div>''')
        #     translated_chatbot_tl = gr.Chatbot([], elem_id="translated_chatbot_tl", height=350)
    
    # gr.Markdown('''<div style="text-align:center"><h2>CAG</h2></div>''')
    with gr.Row():
        with gr.Column(scale=0.33):
            gr.Markdown('''<div style="text-align:center"><h2>華文</h2></div>''')
            cag_chatbot = gr.Chatbot([], elem_id="cag_chatbot", height=350)
        with gr.Column(scale=0.33):
            gr.Markdown('''<div style="text-align:center"><h2>台漢</h2></div>''')
            cag_hl_chatbot = gr.Chatbot([], elem_id="cag_hl_chatbot", height=350)
        with gr.Column(scale=0.33):
            gr.Markdown('''<div style="text-align:center"><h2>台羅</h2></div>''')
            cag_tl_chatbot = gr.Chatbot([], elem_id="cag_tl_chatbot", height=350)

    # TTS 播放區域
    with gr.Row():
        with gr.Column(scale=0.7):
            tts_audio = gr.Audio(label="台羅語音", type="filepath", interactive=False)
        with gr.Column(scale=0.3):
            tts_btn = gr.Button('🔊 播放台羅語音', variant='secondary', size='sm')
    
    with gr.Row():
        with gr.Column(scale=0.2, variant='panel'):
            augmented_mode = gr.Dropdown(
                choices=["cag-長照問答", "rag-中西藥併用"],
                value="cag-長照問答",
                label="Select mode",
                type='index',
            )
        
    txt = gr.Textbox(label="Question", lines=2, placeholder="Enter your question and press \"shift+enter\" ")

    with gr.Row():
        with gr.Column(scale=0.5):
            submit_btn = gr.Button('Submit', variant='primary', size='sm')
        with gr.Column(scale=0.5):
            clear_btn = gr.Button('Clear', variant='stop', size='sm')

    augmented_mode.change(mode_change, augmented_mode,[cag_chatbot, cag_hl_chatbot, cag_tl_chatbot])

    txt.submit(
        add_text, [cag_chatbot, txt], [cag_chatbot, txt]
    ).then(
        bot_answer,
        [augmented_mode, cag_chatbot, cag_hl_chatbot, cag_tl_chatbot],
        [cag_chatbot, cag_hl_chatbot, cag_tl_chatbot],queue=True
    ).then(
        clear_cuda_cache, None, None
    )
        # .then(bot, [chatbot], [chatbot, translated_chatbot])
    # ,translated_chatbot_tl

    submit_btn.click(
        add_text, [ cag_chatbot, txt], [cag_chatbot, txt]
    ).then(
        bot_answer,
        [augmented_mode, cag_chatbot, cag_hl_chatbot, cag_tl_chatbot], 
        [cag_chatbot, cag_hl_chatbot, cag_tl_chatbot],queue=True
    ).then(
        clear_cuda_cache, None, None
    )# ,translated_chatbot_tl
    #.then(bot, [chatbot], [chatbot, translated_chatbot])
    # clear_btn.click(lambda: None, None, chatbot, queue=False)
    # clear_btn.click(lambda: None, None, translated_chatbot, queue=False)
    # clear_btn.click(lambda: None, None, translated_chatbot_tl, queue=False)
    clear_btn.click(lambda: None, None, cag_chatbot, queue=False)
    clear_btn.click(lambda: None, None, cag_hl_chatbot, queue=False)
    clear_btn.click(lambda: None, None, cag_tl_chatbot, queue=False)
    clear_btn.click(lambda: None, None, tts_audio, queue=False)

if __name__ == '__main__':
    # demo.queue(concurrency_count=3)
    
    # 掛gradio伺服器
    #demo.launch(debug=True, share=True)
    demo.queue().launch(debug=False, server_name='0.0.0.0', server_port=6006, share=True)
