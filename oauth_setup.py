"""
Script único para obtener el GOOGLE_REFRESH_TOKEN de Eduardo.

Flujo:
1. Abre el navegador en la pantalla de consentimiento de Google.
2. Eduardo autoriza el acceso al calendario.
3. El script captura el code en http://localhost:8080 y lo intercambia.
4. Imprime el refresh_token para pegarlo en Railway como GOOGLE_REFRESH_TOKEN.

Correr una sola vez en la laptop de Eduardo:

    export GOOGLE_CLIENT_ID="..."
    export GOOGLE_CLIENT_SECRET="..."
    pip install google-auth-oauthlib
    python3 oauth_setup.py

Requisitos previos en Google Cloud Console:
- El OAuth client debe ser de tipo "Desktop app" (o "Web" con
  http://localhost:8080 agregado en Authorized redirect URIs).
- La API Google Calendar API debe estar habilitada en el proyecto.
- Si el OAuth app está en modo "Testing", agregar el correo de Eduardo
  como Test user.
"""

import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main() -> None:
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        print(
            "ERROR: faltan variables de entorno.\n"
            "Exporta GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET antes de correr.",
            file=sys.stderr,
        )
        sys.exit(1)

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8080/"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    # access_type=offline + prompt=consent garantiza refresh_token en la respuesta.
    creds = flow.run_local_server(
        port=8080,
        access_type="offline",
        prompt="consent",
        open_browser=True,
    )

    if not creds.refresh_token:
        print(
            "⚠️  Google no devolvió refresh_token. Esto pasa si antes ya diste "
            "consent con el mismo scope. Revoca el acceso en "
            "https://myaccount.google.com/permissions y vuelve a correr el script.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("\n═══════════════════════════════════════════════════════════════")
    print("  GOOGLE_REFRESH_TOKEN — copia esto a Railway:")
    print("───────────────────────────────────────────────────────────────")
    print(creds.refresh_token)
    print("═══════════════════════════════════════════════════════════════\n")
    print("Comando Railway:")
    print(f'  railway variables --set "GOOGLE_REFRESH_TOKEN={creds.refresh_token}"')
    print()


if __name__ == "__main__":
    main()
