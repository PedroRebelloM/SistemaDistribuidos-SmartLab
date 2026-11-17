import os
import time
import logging
import threading
import requests
import cv2

URL = "http://127.0.0.1:8000"

ESTADO_BUSCANDO = "BUSCANDO_TAG"
ESTADO_EXECUTANDO = "EXECUTANDO_ACAO"
ESTADO_ESTABILIZANDO = "ESTABILIZANDO"

# Parâmetros da espera de chegada
TOLERANCIA_CHEGADA = 0.3   # metros
TIMEOUT_CHEGADA = 20       # segundos
INTERVALO_CHEGADA = 0.5    # segundos entre cada consulta de telemetria

ARQUIVO_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missao.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(ARQUIVO_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("missao")


class ControladorMissao:
    def __init__(self):
        self.estado = ESTADO_BUSCANDO
        self.ultimoIdExecutado = None
        self.droneEmMovimento = False

        # variáveis para confirmar a leitura (debounce)
        self.idCandidato = None
        self.contagemFrames = 0
        self.FRAMES_CONFIRMACAO = 5

        self.tempoFimAcao = 0

        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def atualizar(self, frame):
        corners, ids = self.detectarAruco(frame)
        self.desenharOverlay(frame, corners, ids)

        if self.estado == ESTADO_BUSCANDO:
            if self.droneEmMovimento:
                return

            if ids is not None and len(ids) > 0:
                tagAtual = int(ids.flatten()[0])

                if tagAtual == self.idCandidato:
                    self.contagemFrames += 1
                else:
                    self.idCandidato = tagAtual
                    self.contagemFrames = 1

                if self.contagemFrames >= self.FRAMES_CONFIRMACAO:
                    self.iniciarAcao(tagAtual)
            else:
                self.idCandidato = None
                self.contagemFrames = 0

        elif self.estado == ESTADO_EXECUTANDO:
            if not self.droneEmMovimento:
                log.info("Movimento concluído. Estabilizando")
                self.tempoFimAcao = time.time()
                self.estado = ESTADO_ESTABILIZANDO

        elif self.estado == ESTADO_ESTABILIZANDO:
            if time.time() - self.tempoFimAcao >= 1.0:
                log.info("Pronto para ler próxima tag.")
                self.estado = ESTADO_BUSCANDO

    def iniciarAcao(self, tagId):
        acao = TAG_ACOES.get(tagId)
        if acao is None:
            log.warning(f"Tag {tagId} confirmada, mas não há ação cadastrada. Ignorando.")
            self.contagemFrames = 0
            return

        log.info(f"Tag {tagId} confirmada. Ação: {acao.__name__}")
        self.estado = ESTADO_EXECUTANDO
        self.ultimoIdExecutado = tagId
        self.contagemFrames = 0
        self.droneEmMovimento = True

        thread = threading.Thread(target=self._executarComandoApi, args=(tagId, acao))
        thread.start()

    def _executarComandoApi(self, tagId, acao):
        """Roda em thread separada e libera o estado quando a API responder."""
        try:
            acao()
        except Exception as e:
            log.error(f"Erro ao executar {acao.__name__} (Tag {tagId}): {e}")
        finally:
            self.droneEmMovimento = False

    def detectarAruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def desenharOverlay(self, frame, corners, ids):
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.putText(frame, f"Estado: {self.estado}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.estado == ESTADO_BUSCANDO and self.idCandidato is not None:
            texto = f"Tag {self.idCandidato}: {self.contagemFrames}/{self.FRAMES_CONFIRMACAO}"
            cv2.putText(frame, texto, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.ultimoIdExecutado is not None:
            acao = TAG_ACOES.get(self.ultimoIdExecutado)
            nome = acao.__name__ if acao else "?"
            cv2.putText(frame, f"Ultima: Tag {self.ultimoIdExecutado} ({nome})", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def getPosicao():
    resposta = requests.get(f"{URL}/telemetry/ned", timeout=5)
    pos = resposta.json()["info"]["position"]
    return pos["x"], pos["y"], pos["z"]


def aguardarChegada(alvo, tolerancia=TOLERANCIA_CHEGADA, timeout=TIMEOUT_CHEGADA,
                    intervalo=INTERVALO_CHEGADA):
    """Consulta a telemetria (com sleep entre consultas) até chegar ao alvo ou estourar o timeout."""
    inicio = time.time()
    while time.time() - inicio < timeout:
        x, y, z = getPosicao()
        distancia = ((x - alvo[0]) ** 2 + (y - alvo[1]) ** 2 + (z - alvo[2]) ** 2) ** 0.5
        if distancia <= tolerancia:
            log.info(f"Chegou ao alvo (erro {distancia:.2f}m)")
            return True
        time.sleep(intervalo)

    log.warning(f"Timeout de {timeout}s aguardando chegada ao alvo {alvo}")
    return False


def arm():
    log.info("Armando motores")
    resArm = requests.get(f"{URL}/command/arm", timeout=10)
    log.info(f"Status arm: {resArm.status_code} {resArm.text}")
    time.sleep(4)


def takeOff(altitude=3):
    log.info(f"Iniciando voo (altitude {altitude}m)")
    resTakeoff = requests.get(f"{URL}/command/takeoff", params={"alt": altitude}, timeout=15)
    log.info(f"Status takeoff: {resTakeoff.status_code} {resTakeoff.text}")
    time.sleep(8)


def land():
    log.info("Pousando")
    resLand = requests.get(f"{URL}/command/land", timeout=10)
    log.info(f"Status land: {resLand.status_code} {resLand.text}")


def drive(x=0, y=0, z=0, lookAtTarget=True):
    """Move o drone de forma relativa (NED) e aguarda a chegada: x = frente/trás, y = direita/esquerda."""
    xAtual, yAtual, zAtual = getPosicao()
    alvo = (xAtual + x, yAtual + y, zAtual + z)
    log.info(f"Posição atual: ({xAtual:.2f}, {yAtual:.2f}, {zAtual:.2f}) -> alvo: "
             f"({alvo[0]:.2f}, {alvo[1]:.2f}, {alvo[2]:.2f})")

    corpoRequisicao = {
        "x": x,
        "y": y,
        "z": z,
        "look_at_target": lookAtTarget
    }
    respostaDrive = requests.post(f"{URL}/movement/drive", json=corpoRequisicao, timeout=10)
    log.info(f"Status drive: {respostaDrive.status_code} {respostaDrive.text}")

    aguardarChegada(alvo)


def driveFront(distancia=2):
    log.info(f"Movendo o drone {distancia}m para frente")
    drive(x=distancia)


def driveBack(distancia=2):
    log.info(f"Movendo o drone {distancia}m para trás")
    drive(x=-distancia)


def driveLeft(distancia=2):
    log.info(f"Movendo o drone {distancia}m para a esquerda")
    drive(y=-distancia)


def driveRight(distancia=2):
    log.info(f"Movendo o drone {distancia}m para a direita")
    drive(y=distancia)


# Mapeia o ID da tag para a função executada
TAG_ACOES = {
    0: arm,
    1: land,
    2: takeOff,
    3: driveFront,
    4: driveBack,
    5: driveLeft,
    6: driveRight,
}


def main():
    controlador = ControladorMissao()
    cap = cv2.VideoCapture(0)

    log.info("=== Missão iniciada ===")
    arm()
    takeOff()

    log.info("Iniciando leitura de ArUco. Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Falha ao capturar frame da câmera.")
            break

        controlador.atualizar(frame)

        cv2.imshow("Visao Drone - ArUco", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("=== Missão encerrada ===")


if __name__ == "__main__":
    main()
