import cv2 as cv
import numpy as np
import time
import os

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(os.path.dirname(dir_actual))
carpeta_data_raw = os.path.join(dir_raiz, "data", "raw")


def get_energia(imagen):
    imagen = imagen.astype(np.float32)
    return np.mean(imagen**2)


def ver_por_frame(video, energias, interrupciones, referencias):
    upper_cap_slider_max = len(video) - 1

    title_frame = "Frame"
    title_ref = "Referencia"
    title_diff = "Diferencia"

    def on_trackbar(x):
        frame = video[x].copy()
        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_ref = referencias[x]
        diff = cv.absdiff(gris, frame_ref)

        energia = energias[x]
        interrupcion = interrupciones[x]

        cv.putText(frame, f"Frame: {x}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Energia: {energia:.2f}", (20, 65), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Interrupcion: {interrupcion}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if interrupcion else (0, 255, 0), 2)

        cv.imshow(title_frame, frame)
        cv.imshow(title_ref, frame_ref)
        cv.imshow(title_diff, diff)

    cv.namedWindow(title_frame)
    cv.namedWindow(title_ref)
    cv.namedWindow(title_diff)

    cv.createTrackbar("Frame", title_frame, 0, upper_cap_slider_max, on_trackbar)
    on_trackbar(0)

    cv.waitKey()
    cv.destroyAllWindows()


def ejecutar_foto_captura(vivo=True, url='', ms=500, actualizar_ref_ms=5000, N_estables=5, umbral=1500):
    cap = cv.VideoCapture(0 if vivo else url)

    if not cap.isOpened():
        print(f"Error: No se pudo encontrar o leer el video en: {url}")
        return [], [], [], []
    
    video = []
    energias = []
    interrupciones = []
    referencias = []

    frame_ref = None

    if vivo:
        ultimo_t = 0
        ultimo_refresco = 0
        frames_estables = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ahora = time.perf_counter()

            if (ahora - ultimo_t) * 1000 >= ms:
                ultimo_t = ahora
                video.append(frame)
                gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                if frame_ref is None:
                    frame_ref = gris.copy()
                    ultimo_refresco = ahora

                referencias.append(frame_ref.copy())
                diff = cv.absdiff(gris, frame_ref)
                energia = get_energia(diff)
                energias.append(energia)

                interrupcion = energia > umbral
                interrupciones.append(interrupcion)

                if interrupcion:
                    frames_estables = 0
                else:
                    frames_estables += 1

                tiempo_desde_refresco = (ahora - ultimo_refresco) * 1000

                if (not interrupcion and tiempo_desde_refresco >= actualizar_ref_ms and frames_estables >= N_estables):
                    frame_ref = gris.copy()
                    ultimo_refresco = ahora
                    frames_estables = 0
                    print(f"Referencia actualizada ({N_estables} frames estables)")

                print(f"E={energia:.2f} | Interrupcion={interrupcion} | Frames estables={frames_estables}")

            cv.imshow("En vivo", frame)
            if cv.waitKey(1) & 0xFF == ord('c'):
                break

    else:
        t = 0
        ultimo_refresco_ms = 0
        frames_estables = 0

        while True:
            cap.set(cv.CAP_PROP_POS_MSEC, t)
            ret, frame = cap.read()
            if not ret:
                break

            video.append(frame)
            gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if frame_ref is None:
                frame_ref = gris.copy()

            referencias.append(frame_ref.copy())
            diff = cv.absdiff(gris, frame_ref)
            energia = get_energia(diff)
            energias.append(energia)

            interrupcion = energia > umbral
            interrupciones.append(interrupcion)

            if interrupcion:
                frames_estables = 0
            else:
                frames_estables += 1

            if (not interrupcion and t - ultimo_refresco_ms >= actualizar_ref_ms and frames_estables >= N_estables):
                frame_ref = gris.copy()
                ultimo_refresco_ms = t
                frames_estables = 0
                print(f"Referencia actualizada en {t} ms ({N_estables} frames estables)")

            t += ms

    cap.release()
    cv.destroyAllWindows()

    return video, energias, interrupciones, referencias

if __name__ == '__main__':
    
    ruta = os.path.join(carpeta_data_raw, "Prueba1.mp4")

    video, energias, interrupciones, referencias = ejecutar_foto_captura(vivo=False, url = ruta, ms=500, actualizar_ref_ms=2000, N_estables=3, umbral=500)

    ver_por_frame(video, energias, interrupciones, referencias)