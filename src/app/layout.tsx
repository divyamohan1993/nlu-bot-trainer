import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NLU Bot Trainer",
  description:
    "Train and test Natural Language Understanding models for chatbots with a visual interface",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen bg-surface-0 text-gray-200">
        <a href="#main-content" className="skip-nav">
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
