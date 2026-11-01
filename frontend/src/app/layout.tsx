import type { Metadata } from "next";
import { Cabin, Fira_Mono, Chivo, Inter, Doto } from "next/font/google";
import "./globals.css";

const cabin = Cabin({ variable: "--font-cabin", subsets: ["latin"] });
const firaMono = Fira_Mono({ variable: "--font-fira", subsets: ["latin"], weight: ["400", "500", "700"] });
const chivo = Chivo({ variable: "--font-chivo", subsets: ["latin"] });
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const doto = Doto({ variable: "--font-doto", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ClaustroScope // Psychological Perception Engine",
  description:
    "Experience how claustrophobia feels. Multi-modal spatial perception and Multi-ControlNet diffusion simulating acute enclosure and dread.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${cabin.variable} ${firaMono.variable} ${chivo.variable} ${inter.variable} ${doto.variable} antialiased bg-background text-foreground font-inter`}
      >
        {children}
      </body>
    </html>
  );
}
