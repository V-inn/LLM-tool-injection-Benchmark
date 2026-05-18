# Salve como: worker_node.py
import socket
import time

def start_worker():
    UDP_IP = "0.0.0.0" # Escuta em todas as interfaces da rede
    UDP_PORT = 5005    # Porta arbitrária escolhida para a nossa rede secreta

    # Cria o socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"[*] Worker Node iniciado e escutando na porta UDP {UDP_PORT}...")
    print("[*] Aguardando o Master Controller chamar...")

    while True:
        # Fica travado aqui até receber uma mensagem
        data, addr = sock.recvfrom(1024) 
        mensagem = data.decode('utf-8')
        
        master_ip = addr[0]

        if mensagem == "OLLAMA_MASTER_SEEKING":
            print(f"[!] Grito recebido do Master no IP: {master_ip}")
            print(f"[+] Enviando confirmação de prontidão...")
            
            # Responde diretamente para o IP do Master
            resposta = "OLLAMA_READY"
            sock.sendto(resposta.encode('utf-8'), addr)
            
            # Pequeno delay para não fludar a rede se houver muitos broadcasts
            time.sleep(1)

if __name__ == "__main__":
    start_worker()