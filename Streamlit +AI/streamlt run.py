from pyngrok import ngrok
import os
from google.colab import userdata
from config import NGROK_AUTH_TOKEN


try:
    ngrok_auth_token = userdata.get('NGROK_AUTH_TOKEN')
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)
        print("Ngrok authtoken set from Colab secrets.")
    else:
        print("Ngrok authtoken not found in Colab secrets. Please add it.")
        # You can also set it directly like this, but it's not recommended for security:
        ngrok.set_auth_token("token")

except Exception as e:
    print(f"Error retrieving ngrok authtoken from Colab secrets: {e}")
    # Fallback to setting it directly if needed, but keep YOUR_NGROK_AUTH_TOKEN secure
    ngrok.set_auth_token("Token")
    print("Attempting to proceed without authtoken (may fail if not set elsewhere).")


# Kill previous tunnels
print("Killing previous ngrok tunnels...")
ngrok.kill()
print("Previous tunnels killed.")

# Define the port Streamlit will use
streamlit_port = 8501

# Create tunnel to the Streamlit port
print(f"Creating ngrok tunnel to port {streamlit_port}...")
try:
    public_url = ngrok.connect(streamlit_port)
    print("Streamlit URL:", public_url)

    # Run Streamlit app in the background
    print(f"Running Streamlit app on port {streamlit_port}...")
    # Use a different command to run the app in the background and redirect output
    # !streamlit run app.py --server.port {streamlit_port} > /dev/null 2>&1 &
    # Using %%bash magic command to run in background more reliably
    get_ipython().system_raw(f'streamlit run app.py --server.port {streamlit_port} &')


except Exception as e:
    print(f"❌ Error creating ngrok tunnel or running Streamlit: {e}")
    print("Please ensure you have installed streamlit (`!pip install streamlit`)")
    print("and that 'app.py' exists and is runnable.")