import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinAudit AI — Evidence-Linked Financial Analysis",
  description: "Evidence-linked financial audit intelligence, report discovery, deterministic metric extraction, and anomaly detection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
