import cv2
import os

def main():
    ruta_video = '../../../data/raw/Prueba1.mp4'
    if not os.path.exists(ruta_video):
        print(f"❌ Error: No se encontró el archivo en la ruta: {os.path.abspath(ruta_video)}")
        return
    
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir el archivo de video.")
        return

    print("▶️ Reproduciendo video en escala de grises...")
    print("⌨️ Presioná la tecla 'q' con la ventana del video seleccionada para salir.")

    while cap.isOpened(): #bucle principal
        ret, frame = cap.read()
        
        if not ret: #si se terminan los fotogramas salir
            print("🏁 Fin del video.")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertimos el programa en escala de grises
        cv2.imshow('Visualizador de Ajedrez - B&N', gray_frame)#mostramos en pantalla
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("🛑 Reproducción detenida por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🧹 Recursos liberados y ventanas cerradas correctamente.")

if __name__ == '__main__':
    main()