import cv2
import os
from deepface import DeepFace



# Cria uma função robusta para encontrar a coluna correta de distância, sanando o erro o cóigo anterior

def encontrar_coluna_distancia(match, model_name, distance_metric):
    """
    Retorna o nome correto da coluna de distância, independentemente da versão do DeepFace.
    """

    candidates = [
        f"{distance_metric}_{model_name}",
        f"{distance_metric}_{model_name.lower()}",
        f"{distance_metric}_{model_name.replace('-', '_')}",

        f"{model_name}_{distance_metric}",
        f"{model_name.lower()}_{distance_metric}",
        f"{model_name.replace('-', '_')}_{distance_metric}",

        "distance",
        "threshold"
    ]

    # verificar se alguma existe
    for c in candidates:
        if c in match.index:
            return c

    # fallback: qualquer coluna contendo "cosine" ou "euclidean"
    for col in match.index:
        if distance_metric in col.lower():
            return col

    return None


# Configurações - instanciamento

model_name = "VGG-Face"
distance_metric = "cosine"
db_path = "faces_db"   # pasta com fotos cadastradas


# Iniciar webcam

cap = cv2.VideoCapture(0)

print("🎥 Sistema iniciado! Pressione 'q' para sair.\n")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar webcam.")
        break

    # tentar reconhecer rosto na imagem
    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=db_path,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False
        )

        if isinstance(results, list) and len(results) > 0:
            df = results[0]

            if df.shape[0] > 0:
                match = df.iloc[0]  # melhor correspondência

                # encontrar column correta
                distance_key = encontrar_coluna_distancia(match, model_name, distance_metric)

                if distance_key:
                    distancia = match[distance_key]
                    identity_path = match["identity"]
                    nome_pessoa = os.path.splitext(os.path.basename(identity_path))[0]

                    texto = f"{nome_pessoa} ({distancia:.3f})"
                else:
                    texto = "Distância não encontrada"
            else:
                texto = "Desconhecido"
        else:
            texto = "Desconhecido"

    except Exception as e:
        texto = f"Erro: {str(e)}"

    # exibir nome no frame
    cv2.putText(frame, texto, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento Facial", frame)

    # fechar com K o programa
    if cv2.waitKey(1) & 0xFF == ord("k"):
        break


cap.release()
cv2.destroyAllWindows()
