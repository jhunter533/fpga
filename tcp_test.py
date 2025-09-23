import asyncio
import socket

async def test_fpga_connection():
    host='192.168.2.5'
    port=8888
    try:
        reader,writer=await asyncio.open_connection(host,port)
        print(f"Connected to fpga at {host}:{port}")
        message="helllo world"
        print(f"Sent: {message}")
        writer.write(message.encode('utf-8'))
        await writer.drain()
        response = await reader.read(1024)
        print(f"Received: {response.decode('utf-8')}")
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        print(f"Error {e}")
if __name__=="__main__":
    asyncio.run(test_fpga_connection())
