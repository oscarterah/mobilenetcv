import Felgo 3.0
import QtQuick 2.12
import QtQuick.Controls 2.12
import QtMultimedia 5.13
import CVFilter 1.0

App {
    id: root
//    width: 900
//    height: 600

    property bool drawing: false

    function resetBoundingBoxes() {
        for (var i = 0; i < boundingBoxesHolder.count; ++i)
            boundingBoxesHolder.itemAt(i).visible = false;
    }


    Timer{
        id: drawingTimer
        interval: 30
        onTriggered: {
            drawing = false;
        }
    }

    Camera {
        id: camera
        position: Camera.BackFace
        viewfinder {
            //resolution: "320x240"
            maximumFrameRate: 30
        }
    }

    CVFilter{
        id: cvFilter	
        onObjectDetected: {

            if(drawing) return;

            drawing = true;

            resetBoundingBoxes();

            rects = JSON.parse(rects);

            mj = JSON.parse(mj);
            for(var i in mj)
            {
//                labels.text = mj[i];
                console.log(mj[i]);
            }

            var contentRect = output.contentRect;

            for(let i = 0; i < rects.length; i++){

                var boundingBox = boundingBoxesHolder.itemAt(i);

                var r = {
                    x: rects[i].rX * contentRect.width,
                    y: rects[i].rY * contentRect.height,
                    width: rects[i].rWidth * contentRect.width,
                    height: rects[i].rHeight * contentRect.height
                };

                boundingBox.x = r.x;
                boundingBox.y = r.y;
                boundingBox.width = r.width;
                boundingBox.height = r.height;
                boundingBox.visible = true;

            }

            drawingTimer.start();

        }
    }

    VideoOutput {
        id: output
        source: camera
        anchors.fill: parent
        focus : visible
        fillMode: VideoOutput.PreserveAspectCrop
        filters: [cvFilter]
        autoOrientation: true

        //bounding boxes parent
        Item {
            width: output.contentRect.width
            height: output.contentRect.height
            anchors.centerIn: parent

            Repeater{
                id: boundingBoxesHolder
                model: 20

                Rectangle{
                    border.width: 7
                    border.color: "#146db3"
                    visible: false
                    color: "transparent"
                    radius: 30

                    Text{
                        id: labels
                        text: qsTr("terah")
                        anchors.left : parent.left
                        anchors.leftMargin: 12
                        anchors.bottom : parent.bottom
                        anchors.bottomMargin: 12
                        color: "#FFFFFF"
                        font.pointSize: 20
                        font.bold: true
                        font.family: "Arial"
                    }
                }

            }
        }

    }
}
