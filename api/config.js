export const config = {
  runtime: 'edge',
};

export default function handler(req) {
  const primaryEngineUrl =
    process.env.PRIMARY_ENGINE_URL ||
    process.env.NEXT_PUBLIC_PRIMARY_ENGINE_URL ||
    '';
  const kaggleTunnelUrl = process.env.KAGGLE_TUNNEL_URL || '';

  return new Response(
    JSON.stringify({
      primary_engine_url: primaryEngineUrl,
      kaggle_tunnel_url: kaggleTunnelUrl,
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-store',
      },
    }
  );
}
