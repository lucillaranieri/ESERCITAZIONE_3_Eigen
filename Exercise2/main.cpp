//ESERCITAZIONE_3_Eigen

#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace Eigen;
using namespace std;

// Funzione per risolvere il sistema lineare e calcolare l'errore relativo
void risEVer(const Matrix2d& A, const Vector2d& b, const Vector2d& x_esatto) {
    // Decomposizione PALU (Permutazione, Matrice Inferiore e Matrice Superiore)
    FullPivLU<Matrix2d> lu(A);
    Vector2d x_palu = lu.solve(b); // Risoluzione del sistema
    double errore_palu = (x_palu - x_esatto).norm() / x_esatto.norm(); // Calcolo dell'errore relativo

    // Decomposizione QR (Fattorizzazione QR con riflessioni di Householder)
    HouseholderQR<Matrix2d> qr(A);
    Vector2d x_qr = qr.solve(b); // Risoluzione del sistema
    double errore_qr = (x_qr - x_esatto).norm() / x_esatto.norm(); // Calcolo dell'errore relativo

    // Stampa dei risultati
    cout << "La soluzione del metodo PALU è: " << scientific << setprecision(20) << x_palu.transpose() << endl;
    cout << "L'errore relativo è (PALU): " << scientific << setprecision(20) << errore_palu << endl;
    
    cout << "La soluzione del metodo QR è: " << scientific << setprecision(20) << x_qr.transpose() << endl;
    cout << "L'errore relativo è (QR): " << scientific << setprecision(20) << errore_qr << endl;
    cout << "-----------------------------" << endl;
}

int main() {
    // Soluzione esatta attesa
    Vector2d x_esatto(-1.0, -1.0);

    // Primo sistema
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
    risEVer(A1, b1, x_esatto);

    // Secondo sistema
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);
    risEVer(A2, b2, x_esatto);

    // Terzo sistema
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);
    risEVer(A3, b3, x_esatto);

    return 0;
}
