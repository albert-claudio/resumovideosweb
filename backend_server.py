# backend_server.py
import os
import tempfile
import subprocess
import json
import requests
import re
import whisper
import torch # Whisper pode depender de torch

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Para permitir requisições do React

# Importações para ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Carregar variáveis de ambiente do arquivo .env, se existir
# É bom chamar isso no início para que a GEMINI_API_KEY esteja disponível
load_dotenv()

# --- Funções do seu script app.py ---
# (Copiadas diretamente do seu arquivo, com pequenos ajustes nos prints para logging/feedback do servidor)

def limpar_nome_ficheiro(url): #
    """
    Cria um nome de ficheiro seguro a partir de uma URL ou título.
    """
    try:
        # Tenta extrair o ID do vídeo para um nome mais curto
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url) #
        if match:
            nome_base = match.group(1) #
        else:
            # Se não for uma URL do YouTube reconhecível, usa uma parte da URL
            nome_base = url.split('/')[-1] if '/' in url else url #
            nome_base = nome_base.split('?')[0] # Remove parâmetros query #
        
        # Remove caracteres inválidos para nomes de ficheiro
        nome_seguro = re.sub(r'[\\/*?:"<>|]', "", nome_base) #
        nome_seguro = nome_seguro[:50] # Limita o comprimento #
        return f"resumo_video_{nome_seguro}" #
    except Exception:
        return "resumo_video_desconhecido" #


def baixar_audio_youtube_yt_dlp(url, output_dir): #
    """
    Baixa a melhor stream de áudio de um vídeo do YouTube usando yt-dlp.
    Retorna o caminho do ficheiro de áudio baixado ou None em caso de erro.
    """
    try:
        print(f"BACKEND: A tentar baixar áudio com yt-dlp de: {url}...")
        try:
            subprocess.run(['yt-dlp', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\nBACKEND ERRO: yt-dlp não foi encontrado ou não está a funcionar corretamente.")
            return None

        audio_format = 'm4a' #
        audio_filename_base = "downloaded_audio_for_summary" #
        # Garantir que o output_filepath seja construído corretamente para o yt-dlp
        # O yt-dlp vai adicionar a extensão, então não a inclua no nome base para --output
        output_template = os.path.join(output_dir, f"{audio_filename_base}.%(ext)s")
        # Caminho esperado após o download
        expected_output_filepath = os.path.join(output_dir, f"{audio_filename_base}.{audio_format}")


        command = [
            'yt-dlp', '--extract-audio', '-x', '--audio-format', audio_format, #
            '--output', output_template, '--no-playlist', '--quiet', '--no-warnings', #
            url
        ]
        
        print(f"BACKEND: A executar comando: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #
        stdout, stderr = process.communicate() #

        if process.returncode == 0: #
            if os.path.exists(expected_output_filepath):
                print(f"BACKEND: Áudio baixado com yt-dlp para: {expected_output_filepath}")
                return expected_output_filepath #
            else:
                # Fallback para verificar se algum arquivo com o nome base foi criado,
                # pois a lógica original do seu script tinha um fallback para `possible_files`.
                # Isso pode acontecer se o yt-dlp salvar com uma pequena variação no nome ou se a extração inicial falhar
                # e ele tentar um formato diferente.
                print(f"BACKEND: yt-dlp executado, mas ficheiro esperado ({expected_output_filepath}) não encontrado diretamente.")
                possible_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(audio_filename_base) and f.endswith(f".{audio_format}")] #
                if possible_files:
                    found_file = possible_files[0] #
                    print(f"BACKEND: Ficheiro de áudio alternativo encontrado: {found_file}")
                    return found_file #
                print(f"BACKEND: Nenhum ficheiro de áudio correspondente encontrado em {output_dir}")
                print(f"BACKEND Stdout: {stdout.decode('utf-8', errors='ignore')}")
                print(f"BACKEND Stderr: {stderr.decode('utf-8', errors='ignore')}")
                return None
        else:
            print(f"BACKEND: Erro ao baixar o áudio com yt-dlp (código de saída: {process.returncode}):")
            error_message = stderr.decode('utf-8', errors='ignore').strip() or stdout.decode('utf-8', errors='ignore').strip() #
            print(f"BACKEND: Mensagem de erro do yt-dlp: {error_message if error_message else 'Nenhuma mensagem de erro específica.'}")
            return None
    except FileNotFoundError: #
        print("BACKEND ERRO CRÍTICO: O executável 'yt-dlp' não foi encontrado.")
        return None
    except Exception as e:
        print(f"BACKEND: Exceção inesperada ao tentar usar yt-dlp: {e}")
        return None

def transcrever_audio(caminho_audio, modelo_whisper="base"): #
    """
    Transcreve o áudio para texto em Português usando o modelo Whisper.
    Retorna o texto transcrito ou None em caso de erro.
    """
    if not caminho_audio or not os.path.exists(caminho_audio): #
        print(f"BACKEND Erro na transcrição: Caminho do áudio não encontrado ou inválido: {caminho_audio}")
        return None
    try:
        print(f"BACKEND: A carregar modelo Whisper ({modelo_whisper})...")
        device = "cuda" if torch.cuda.is_available() else "cpu" #
        print(f"BACKEND: A usar dispositivo para Whisper: {device}")
        
        model = whisper.load_model(modelo_whisper, device=device) #
        print("BACKEND: Modelo Whisper carregado. A transcrever áudio em Português...")
        
        resultado = model.transcribe(caminho_audio, fp16=torch.cuda.is_available(), language="pt") #
        texto_transcrito = resultado["text"] #
        
        print("BACKEND: Transcrição concluída.")
        if not texto_transcrito.strip(): #
            print("BACKEND: A transcrição resultou em texto vazio.")
            return None
        return texto_transcrito
    except Exception as e:
        print(f"BACKEND: Erro durante a transcrição do áudio: {e}")
        return None

def resumir_texto_com_gemini(texto_transcrito, max_tokens_resumo=250): # # Seu script usa 200 no main, mas a função define 250. Usaremos 250 como padrão aqui.
    """
    Resume o texto transcrito usando a API Gemini.
    Retorna o texto resumido ou None em caso de erro.
    """
    if not texto_transcrito or not texto_transcrito.strip(): #
        print("BACKEND: Nenhum texto para resumir.")
        return None

    print("BACKEND: A iniciar sumarização com a API Gemini...")
    
    api_key = os.getenv("GEMINI_API_KEY") #
    if not api_key:
        print("BACKEND ERRO: GEMINI_API_KEY não configurada nas variáveis de ambiente.")
        return None
        
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}" #

    prompt_instrucao = (
        f"Você é um assistente especializado em resumir textos de forma concisa e precisa em português brasileiro.\n"
        f"Resuma o seguinte texto, extraindo as informações mais importantes. "
        f"Tente manter o resumo com aproximadamente {int(max_tokens_resumo * 0.6)} a {max_tokens_resumo} palavras, " #
        f"mas priorize a qualidade e a cobertura dos pontos chave sobre a contagem exata de palavras.\n\n"
        f"Texto a ser resumido:\n\"\"\"\n{texto_transcrito}\n\"\"\""  #
    ) 

    chat_history = [{"role": "user", "parts": [{"text": prompt_instrucao}]}] #
    payload = {"contents": chat_history} #
    
    headers = {'Content-Type': 'application/json'} #

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180) #
        response.raise_for_status() #
        
        result = response.json() #

        if (result.get("candidates") and #
            result["candidates"][0].get("content") and #
            result["candidates"][0]["content"].get("parts") and #
            result["candidates"][0]["content"]["parts"][0].get("text")): #
            
            texto_resumido = result["candidates"][0]["content"]["parts"][0]["text"] #
            print("BACKEND: Resumo gerado pela API Gemini.")
            return texto_resumido.strip() #
        else:
            print("BACKEND Erro: A resposta da API Gemini não continha o texto do resumo esperado.")
            print(f"BACKEND Resposta completa da API: {result}")
            if result.get("promptFeedback"): #
                print(f"BACKEND Feedback do prompt: {result.get('promptFeedback')}") #
            return None

    except requests.exceptions.RequestException as e: #
        print(f"BACKEND Erro na requisição à API Gemini: {e}")
        if hasattr(e, 'response') and e.response is not None: #
            try:
                print(f"BACKEND Detalhes do erro da API: {e.response.json()}") #
            except json.JSONDecodeError: #
                print(f"BACKEND Conteúdo da resposta (não JSON): {e.response.text}") #
        return None
    except Exception as e:
        print(f"BACKEND Erro inesperado ao processar resposta da API Gemini: {e}")
        return None

def gerar_pdf_resumo(texto_resumo, nome_ficheiro_pdf_completo): #
    """
    Gera um ficheiro PDF com o texto do resumo.
    Recebe o caminho completo onde o PDF deve ser salvo.
    """
    if not texto_resumo or not texto_resumo.strip(): #
        print("BACKEND: Nenhum texto de resumo para gerar PDF.")
        return False

    try:
        print(f"BACKEND: A gerar PDF: {nome_ficheiro_pdf_completo}...")
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf')) #
            font_name = 'DejaVuSans' #
        except Exception: 
            font_name = 'Helvetica' #
            print("BACKEND: A usar fonte padrão 'Helvetica' para o PDF (DejaVuSans não encontrada).")

        doc = SimpleDocTemplate(nome_ficheiro_pdf_completo, pagesize=A4, #
                                rightMargin=inch, leftMargin=inch, #
                                topMargin=inch, bottomMargin=inch) #
        styles = getSampleStyleSheet() #
        
        style_titulo = styles['h1'] #
        style_titulo.fontName = font_name #
        style_titulo.alignment = 1 # Centralizado #
        
        style_corpo = styles['Normal'] #
        style_corpo.fontName = font_name #
        style_corpo.fontSize = 11 #
        style_corpo.leading = 14 #
        style_corpo.alignment = 4 # Justificado #

        story = [] #
        
        titulo_texto = "Resumo do Vídeo do YouTube" #
        paragrafo_titulo = Paragraph(titulo_texto, style_titulo) #
        story.append(paragrafo_titulo) #
        story.append(Spacer(1, 0.3*inch)) #

        texto_resumo_formatado = texto_resumo.replace('\n', '<br/>\n') #
        paragrafo_resumo = Paragraph(texto_resumo_formatado, style_corpo) #
        story.append(paragrafo_resumo) #
        
        doc.build(story) #
        print(f"BACKEND: PDF gerado com sucesso: {nome_ficheiro_pdf_completo}")
        return True
    except FileNotFoundError as e: #
        print(f"BACKEND Erro ao gerar PDF: Ficheiro de fonte não encontrado. {e}")
        return False
    except Exception as e:
        print(f"BACKEND Erro ao gerar PDF: {e}")
        return False

# --- Configuração do Flask ---
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir requisições do seu frontend React

# Diretório para salvar os PDFs gerados (pode ser dentro do backend ou um local configurável)
PDF_OUTPUT_DIRECTORY = "generated_pdfs" 
if not os.path.exists(PDF_OUTPUT_DIRECTORY):
    os.makedirs(PDF_OUTPUT_DIRECTORY)

# Verifica FFmpeg no início para dar um aviso se não estiver presente
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("BACKEND: FFmpeg encontrado.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nBACKEND AVISO: FFmpeg não encontrado ou não está a funcionar corretamente.")
        print("FFmpeg é essencial para o processamento de áudio pelo yt-dlp e Whisper.")
        print("Por favor, instale o FFmpeg e certifique-se de que está no PATH do seu sistema.\n")
        return False

ffmpeg_disponivel = check_ffmpeg()


@app.route('/api/summarize', methods=['POST'])
def summarize_video_api():
    if not ffmpeg_disponivel: #
        return jsonify({"error": "FFmpeg não está disponível no servidor. Processamento não pode continuar."}), 503

    data = request.get_json()
    video_url = data.get('url')

    if not video_url:
        return jsonify({"error": "URL do vídeo é obrigatória."}), 400

    print(f"BACKEND: Recebida URL para resumir: {video_url}")

    # Usar diretório temporário para o arquivo de áudio
    with tempfile.TemporaryDirectory() as tmpdir: #
        print(f"BACKEND: Usando diretório temporário para áudio: {tmpdir}")
        caminho_audio = baixar_audio_youtube_yt_dlp(video_url, tmpdir) #

        if not caminho_audio:
            return jsonify({"error": "Falha ao baixar o áudio do vídeo. Verifique a URL e os logs do servidor."}), 500

        texto_transcrito = transcrever_audio(caminho_audio, modelo_whisper="base") #
        
        # O áudio já foi processado, o diretório temporário será limpo automaticamente na saída do 'with'

        if not texto_transcrito:
            return jsonify({"error": "Falha ao transcrever o áudio."}), 500
        
        # Usar o valor de max_tokens_resumo definido na chamada original do main (200)
        texto_resumido = resumir_texto_com_gemini(texto_transcrito, max_tokens_resumo=200) #
        
        if not texto_resumido:
            return jsonify({"error": "Falha ao gerar o resumo com a IA. Verifique a API Key e os logs."}), 500

        # Geração do PDF
        nome_base_ficheiro_pdf = limpar_nome_ficheiro(video_url) #
        nome_ficheiro_pdf_final = f"{nome_base_ficheiro_pdf}.pdf" #
        caminho_completo_pdf = os.path.join(PDF_OUTPUT_DIRECTORY, nome_ficheiro_pdf_final)

        if gerar_pdf_resumo(texto_resumido, caminho_completo_pdf): #
            # O PDF foi gerado, preparamos a URL para download
            # O nome do arquivo é retornado para que o frontend possa usá-lo no link de download
            pdf_download_path = f"/api/download_pdf/{nome_ficheiro_pdf_final}"
            return jsonify({
                "summary": texto_resumido,
                "pdf_filename": nome_ficheiro_pdf_final, # Para o frontend saber o nome
                "pdf_download_url": pdf_download_path    # Caminho relativo para o frontend montar o link
            })
        else:
            # Mesmo se o PDF falhar, ainda podemos retornar o resumo em texto
            return jsonify({
                "summary": texto_resumido,
                "pdf_filename": None,
                "error_pdf": "Resumo gerado, mas falha ao criar o arquivo PDF."
            })

@app.route('/api/download_pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    try:
        # Caminho completo e seguro para o arquivo PDF
        safe_path = os.path.abspath(os.path.join(PDF_OUTPUT_DIRECTORY, filename))
        
        # Validação para evitar Path Traversal
        if not safe_path.startswith(os.path.abspath(PDF_OUTPUT_DIRECTORY)):
            return jsonify({"error": "Acesso negado."}), 403 # Forbidden
        
        if not os.path.isfile(safe_path):
             return jsonify({"error": "Arquivo PDF não encontrado no servidor."}), 404 # Not Found
        
        print(f"BACKEND: Enviando PDF: {safe_path}")
        return send_file(safe_path, as_attachment=True)
    except Exception as e:
        print(f"BACKEND Erro ao tentar enviar o PDF {filename}: {e}")
        return jsonify({"error": f"Não foi possível baixar o PDF: {str(e)}"}), 500


if __name__ == '__main__':
    # Verificar yt-dlp no início do servidor também é uma boa prática
    try:
        subprocess.run(['yt-dlp', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #
        print("BACKEND: yt-dlp encontrado e funcionando.")
    except (subprocess.CalledProcessError, FileNotFoundError): #
        print("\nBACKEND ERRO CRÍTICO: yt-dlp não foi encontrado ou não está a funcionar corretamente no servidor.")
        print("Por favor, instale ou verifique a sua instalação de yt-dlp: 'pip install yt-dlp'")
        print("Certifique-se também de que está no PATH do sistema do servidor.\n")
        # Você pode decidir se quer que o servidor pare aqui, e.g., exit(1)

    # A porta 5001 é usada para não conflitar com a porta padrão do React (3000)
    app.run(debug=True, port=5001)