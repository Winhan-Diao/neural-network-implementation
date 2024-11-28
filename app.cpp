#include <fstream>
#include "app.hpp"

MainPanel::MainPanel(wxWindow *parent): wxPanel(parent) {
    Bind(wxEVT_LEFT_DOWN, &MainPanel::OnMouseLeftDown, this, wxID_ANY);
    Bind(wxEVT_MOTION, &MainPanel::OnMouseMove, this, wxID_ANY);
    Bind(wxEVT_PAINT, &MainPanel::OnPaint, this, wxID_ANY);
    // Bind(wxEVT_PAINT, PaintGridFunctor{this}, wxID_ANY);

    wxButton *buttonClear = new wxButton(this, wxID_ANY, "Clear");
    buttonClear->SetPosition(this->FromDIP(wxPoint(30, 30)));
    buttonClear->SetSize(this->FromDIP(wxSize(200, 80))); 
    auto font = buttonClear->GetFont();
    font.SetPointSize(20);
    buttonClear->SetFont(font);
    buttonClear->Bind(wxEVT_BUTTON, [this](const wxCommandEvent&) {
        std::for_each_n(reinterpret_cast<unsigned char *>(this->bitmap), 28*28, [](unsigned char& v){
            v = static_cast<unsigned char>(0);
        });
        this->Refresh();
    }, wxID_ANY);

    wxButton *buttonLoad = new wxButton(this, wxID_ANY, "Load Model");
    buttonLoad->SetPosition(this->FromDIP(wxPoint(300, 30)));
    buttonLoad->SetSize(this->FromDIP(wxSize(200, 80))); 
    buttonLoad->Bind(wxEVT_BUTTON, [this](const wxCommandEvent&) {
        if (std::ifstream ifs{"mnist-network-sgnexp-v1.dat", std::ios::binary}) {
            std::cout << "Network Loaded" << "\r\n";
            ifs >> this->n;
            this->networkValid = true;
        }
    });
}

CustomFrame::CustomFrame(): wxFrame(nullptr, wxID_ANY, "Nueral Network Testing") {
    MainPanel *mainPanel = new MainPanel(this);
    std::cout << "Hello world" << "\r\n";       //debug
}

bool CustomApp::OnInit() {
    SetProcessDPIAware();
    frame->Show(true);
    return true;
}

int CustomApp::OnExit() {
    return wxApp::OnExit();
}

CustomApp::CustomApp(): frame(new CustomFrame()) {}

wxIMPLEMENT_APP(CustomApp);