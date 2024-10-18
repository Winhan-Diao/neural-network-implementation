#pragma once
#include <wx/wx.h>
#include "network.hpp"
#include "mnist.hpp"

class MainPanel: public wxPanel {
    size_t pressedCount{0};
    unsigned char bitmap[28][28]{0};
    wxCoord xOffset{40};
    wxCoord yOffset{200};
    int gridLength{20};
    Network n{};
    bool networkValid{false};
    std::valarray<double> result = std::valarray<double>(10);

    void OnMouseLeftDown(wxMouseEvent& event) {}

    void OnMouseMove(wxMouseEvent& event) {
        if (event.LeftIsDown() || event.RightIsDown()) {
            wxPoint pos = event.GetPosition();
            int xIndex = (pos.x - xOffset) / gridLength;
            if (xIndex < 0 || xIndex >= 28) 
                return;
            int yIndex = (pos.y - yOffset) / gridLength;
            if (yIndex < 0 || yIndex >= 28) 
                return;
            if (event.LeftIsDown() xor bool(bitmap[yIndex][xIndex] == 255)) {
                bitmap[yIndex][xIndex] = static_cast<unsigned char>(event.LeftIsDown()? 255: 0);
                if (xIndex + 1 < 28 && bitmap[yIndex][xIndex + 1] != 255) bitmap[yIndex][xIndex + 1] = static_cast<unsigned char>(event.LeftIsDown()? 128: 0);
                if (yIndex + 1 < 28 && bitmap[yIndex + 1][xIndex] != 255) bitmap[yIndex + 1][xIndex] = static_cast<unsigned char>(event.LeftIsDown()? 128: 0);
                if (xIndex - 1 >= 0 && bitmap[yIndex][xIndex - 1] != 255) bitmap[yIndex][xIndex - 1] = static_cast<unsigned char>(event.LeftIsDown()? 128: 0);
                if (yIndex - 1 >= 0 && bitmap[yIndex - 1][xIndex] != 255) bitmap[yIndex - 1][xIndex] = static_cast<unsigned char>(event.LeftIsDown()? 128: 0);
                Refresh();
                if (networkValid) {
                    std::valarray<double> in(28*28);
                    std::transform(reinterpret_cast<unsigned char*>(bitmap), reinterpret_cast<unsigned char*>(bitmap) + 28*28, std::begin(in), [](unsigned char& v) -> double {
                        return v / 255.;
                    });
                    // std::valarray<double> classifiedResult = n.run(in);
                    result = n.run(in);
                }
            }
        }

    }
    void OnPaint(wxPaintEvent&) {
        wxPaintDC dc(this);
        dc.SetPen(*wxBLACK_DASHED_PEN);
        for (int i = 0; i < 29; ++i) {
            dc.DrawLine(xOffset, yOffset + gridLength*i, xOffset + gridLength*28, yOffset + gridLength*i);
            dc.DrawLine(xOffset + gridLength*i, yOffset, xOffset + gridLength*i, yOffset + gridLength*28);
        }
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                if (bitmap[i][j] == static_cast<unsigned char>(255)) {
                    dc.SetBrush(*wxBLACK_BRUSH);
                    dc.DrawRectangle(xOffset + gridLength * j, yOffset + gridLength * i, gridLength, gridLength);
                } else if (bitmap[i][j]) {
                    dc.SetBrush(*wxGREY_BRUSH);
                    dc.DrawRectangle(xOffset + gridLength * j, yOffset + gridLength * i, gridLength, gridLength);
                }
            }
        }
        dc.DrawText(wxString{} << getGreatestLabel(result), xOffset + gridLength*1, yOffset + gridLength*32);
        wxString r;
        for (int i = 0; i < 10; ++i) {
            r << i << ": " << result[i] << "\r\n";
        }
        dc.DrawText(r, xOffset + gridLength*14, yOffset + gridLength*32);
    }
public:
    MainPanel(wxWindow *parent);
};


class CustomFrame: public wxFrame {
public:
    CustomFrame();
};

class CustomApp: public wxApp {
    CustomFrame *frame;
    bool OnInit() override;
    int OnExit() override;
public:
    CustomApp();
};

wxDECLARE_APP(CustomApp);
