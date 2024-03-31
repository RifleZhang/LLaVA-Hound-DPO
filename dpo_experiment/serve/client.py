import socket
import json


HOST = '127.0.0.1'  # Localhost
PORT = 7760        # Choose an unused port

"""
query format:
    video_path = "/mnt/bn/liangkeg/data/videollava/eval/MSRVTT_Zero_Shot_QA/videos/all/video7010.mp4"
    query = "what is the video doing?"
    data = {
        "video_path": video_path,
        "query": query,
        'temperature': 0.0,
        'top_p': 1.0,
        'max_new_tokens': 1024,
    }
response format:
    {
        'status': 'success',
        'message': outputs
    }
"""

class ClientAPI:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port

    def send(self, data):
        """
            data: Dict with keys:
                img_path: str, required
                query: str, required
                temperature: float, default 0.7
                top_p: float, default 0.9
                max_new_tokens: int, default 1024
        """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        # if "video_path" not in data or "query" not in data:
        #     raise ValueError("video_path and query are required fields")
        
        # send data
        data_to_send = json.dumps(data).encode('utf-8')
        client_socket.sendall(data_to_send + b'__end__')

        result = b''
        # Receive the response
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            result += data
        #result = client_socket.recv(1024).decode('utf-8')
        result = json.loads(result)
        client_socket.close()

        return result
    
def main(host=HOST, port=PORT):
    client = ClientAPI(host, port)
    video_path = "/mnt/bn/liangkeg/data/videollava/eval/MSRVTT_Zero_Shot_QA/videos/all/video7010.mp4"
    query = "what is the video doing?"
    data = {
        "video_path": video_path,
        "query": query,
        'temperature': 0.1,
        'top_p': 0.9,
        'max_new_tokens': 1024,
    }
    response_message = client.send(data)
    print(response_message)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
